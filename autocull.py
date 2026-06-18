import argparse
import csv
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from grouper import find_images, find_videos, group_by_time, get_timestamp
from analyzer import analyze
from location import get_gps, place_name


def _best_filename(path: Path) -> str:
    ts = get_timestamp(path)
    date_str = ts.strftime("%Y%m%d") if ts else "unknown"
    coords = get_gps(path)
    loc = place_name(*coords) if coords else None
    return f"{date_str}_{loc or 'unknown'}{path.suffix.lower()}"


def _unique_dest(dest_dir: Path, name: str) -> Path:
    dest = dest_dir / name
    if not dest.exists():
        return dest
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 2
    while (dest_dir / f"{stem}_{i}{suffix}").exists():
        i += 1
    return dest_dir / f"{stem}_{i}{suffix}"


def pick_best(group: list[Path], analyses: dict) -> Path:
    from analyzer import SMILE_THRESHOLD

    face_photos = [p for p in group if analyses[p]["has_face"]]

    if not face_photos:
        return max(group, key=lambda p: analyses[p]["blur_score"])

    candidates = face_photos

    open_eyes = [p for p in face_photos if not analyses[p]["eyes_closed"]]
    if open_eyes:
        candidates = open_eyes

    smiling = [p for p in candidates if analyses[p]["smile_score"] >= SMILE_THRESHOLD]
    if smiling:
        candidates = smiling

    return max(candidates, key=lambda p: analyses[p]["blur_score"])


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _find_exact_duplicates(images: list[Path]) -> tuple[list[Path], list[Path]]:
    seen: dict[str, Path] = {}
    unique: list[Path] = []
    dupes: list[Path] = []
    for p in images:
        try:
            h = _file_hash(p)
        except OSError:
            unique.append(p)
            continue
        if h in seen:
            dupes.append(p)
        else:
            seen[h] = p
            unique.append(p)
    return unique, dupes


def _find_perceptual_duplicates(
    images: list[Path],
    analyses: dict,
    threshold: int = 8,
) -> tuple[list[Path], list[Path]]:
    import imagehash
    from PIL import Image

    # Sort by blur_score descending so the sharpest copy is always kept
    sorted_images = sorted(images, key=lambda p: analyses[p]["blur_score"], reverse=True)

    kept_hashes: list = []
    unique: list[Path] = []
    dupes: list[Path] = []

    for p in sorted_images:
        try:
            img = Image.open(p)
            img.thumbnail((512, 512))
            h = imagehash.phash(img)
        except Exception:
            unique.append(p)
            continue
        if any(abs(h - kh) <= threshold for kh in kept_hashes):
            dupes.append(p)
        else:
            kept_hashes.append(h)
            unique.append(p)

    return unique, dupes


def run(
    input_dir: Path,
    output_dir: Path,
    gap: int | None,
    blur_threshold: float,
    mode: str,
    dry_run: bool = False,
    log: bool = False,
    recursive: bool = False,
):
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        return

    best_dir = output_dir / "best"
    rejected_dir = output_dir / "rejected"

    # Exclude best/ and rejected/ subdirs so re-runs don't reprocess already-sorted files
    _excludes = [best_dir, rejected_dir] if output_dir.is_relative_to(input_dir) else None
    images = find_images(input_dir, recursive=recursive, exclude=_excludes)
    if not images:
        print("No images found.")
        return

    print(f"Found {len(images)} images")

    # Pass 1: exact (MD5) deduplication — fast, no analysis needed
    after_exact, exact_dupes = _find_exact_duplicates(images)
    if exact_dupes:
        print(f"  {len(exact_dupes)} exact duplicate(s) removed")

    workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(analyze, after_exact),
            total=len(after_exact),
            desc="Analyzing",
        ))
    all_analyses = dict(zip(after_exact, results))

    if exact_dupes:
        print(f"  {len(exact_dupes)} exact duplicate(s) will be rejected")

    face_counts = {p: all_analyses[p]["face_count"] for p in after_exact}
    groups = group_by_time(after_exact, gap_seconds=gap, use_clip=True, face_counts=face_counts)
    skipped = len(after_exact) - sum(len(g) for g in groups)
    if skipped:
        print(f"Warning: {skipped} image(s) skipped — no EXIF timestamp")
    gap_info = f"{gap}s" if gap is not None else "auto"
    print(f"Grouped into {len(groups)} session(s) [gap={gap_info}]")

    if dry_run:
        print("[DRY-RUN] No files will be moved or copied.\n")
    else:
        print()
        if mode in ("copy", "move"):
            best_dir.mkdir(parents=True, exist_ok=True)
            rejected_dir.mkdir(parents=True, exist_ok=True)

    transfer = shutil.copy2 if mode == "copy" else shutil.move

    kept = 0
    rejected = 0
    log_rows: list[dict] = []
    meta: dict[str, dict] = {}  # dest_filename → analysis (written to .autocull_meta.json)

    def _log(session, path, result, reason, analysis, dest=""):
        if not log:
            return
        row = {"session": session, "filename": path.name, "result": result, "reason": reason, "dest": dest}
        if analysis:
            row.update({k: analysis[k] for k in ("blur_score", "has_face", "eyes_closed", "smile_score", "face_count")})
        else:
            row.update({"blur_score": "", "has_face": "", "eyes_closed": "", "smile_score": "", "face_count": ""})
        log_rows.append(row)

    for p in exact_dupes:
        if not dry_run:
            if mode == "remove":
                os.remove(p)
            else:
                transfer(str(p), str(rejected_dir / p.name))
        print(f"  [skip] {p.name} (exact duplicate)")
        rejected += 1
        _log("", p, "skip", "exact duplicate", None)

    for i, group in enumerate(groups, 1):
        # Perceptual dedupe within each session — keeps sharpest copy
        analyses = {p: all_analyses[p] for p in group}
        session_unique, phash_dupes = _find_perceptual_duplicates(group, analyses, threshold=3)

        for p in phash_dupes:
            if not dry_run:
                if mode == "remove":
                    os.remove(p)
                else:
                    transfer(str(p), str(rejected_dir / p.name))
            print(f"  [skip] {p.name} (perceptual duplicate)")
            rejected += 1
            _log(i, p, "skip", "perceptual duplicate", analyses[p])

        group = session_unique
        analyses = {p: all_analyses[p] for p in group}
        print(f"Session {i} ({len(group)} photo{'s' if len(group) > 1 else ''}):")

        if len(group) == 1:
            p = group[0]
            dest_name = _best_filename(p)
            if not dry_run and mode != "remove":
                dest = _unique_dest(best_dir, dest_name)
                transfer(str(p), str(dest))
                dest_name = dest.name
                meta[dest_name] = {k: analyses[p][k] for k in ("blur_score", "has_face", "eyes_closed", "smile_score", "face_count")}
            print(f"  [keep] {p.name} -> {dest_name}")
            kept += 1
            _log(i, p, "keep", "", analyses[p], dest_name)
            continue

        best = pick_best(group, analyses)

        for p in group:
            a = analyses[p]
            if p == best:
                dest_name = _best_filename(p)
                if not dry_run and mode != "remove":
                    dest = _unique_dest(best_dir, dest_name)
                    transfer(str(p), str(dest))
                    dest_name = dest.name
                    meta[dest_name] = {k: a[k] for k in ("blur_score", "has_face", "eyes_closed", "smile_score", "face_count")}
                print(f"  [keep] {p.name} -> {dest_name}")
                kept += 1
                _log(i, p, "keep", "", a, dest_name)
            else:
                if a["eyes_closed"]:
                    reason = "eyes closed"
                else:
                    reason = "not best"
                if not dry_run:
                    if mode == "remove":
                        os.remove(p)
                    else:
                        transfer(str(p), str(rejected_dir / p.name))
                        meta[p.name] = {k: a[k] for k in ("blur_score", "has_face", "eyes_closed", "smile_score", "face_count")}
                        meta[p.name]["reason"] = reason
                print(f"  [skip] {p.name} ({reason})")
                rejected += 1
                _log(i, p, "skip", reason, a)

    # Move all video files to best/ (no culling for videos)
    videos = find_videos(input_dir, recursive=recursive, exclude=_excludes)
    if videos:
        print(f"\nFound {len(videos)} video(s) — moving to best/")
        for v in videos:
            dest_name = v.name
            if not dry_run:
                best_dir.mkdir(parents=True, exist_ok=True)
                dest = _unique_dest(best_dir, v.name)
                transfer(str(v), str(dest))
                dest_name = dest.name
            print(f"  [keep] {v.name} -> {dest_name} (video)")
            kept += 1
            _log("video", v, "keep", "video", None, dest_name)

    if not dry_run and mode != "remove" and meta:
        with open(output_dir / ".autocull_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    suffix = " [DRY-RUN]" if dry_run else ""
    print(f"\nDone{suffix} - kept: {kept}, rejected: {rejected}")

    if log and log_rows:
        log_path = output_dir / "autocull_log.csv"
        if not dry_run:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
                writer.writeheader()
                writer.writerows(log_rows)
            print(f"Log saved: {log_path}")
        else:
            print(f"[DRY-RUN] Log would be saved to: {log_path}")

    if not dry_run:
        print(f"Results in: {output_dir}")


def organize_by_location(best_dir: Path) -> dict[str, int]:
    """Move files in best_dir into location-named subdirectories using GPS."""
    from grouper import find_images
    images = find_images(best_dir, recursive=False)
    counts: dict[str, int] = {}
    for p in images:
        coords = get_gps(p)
        loc = place_name(*coords) if coords else "unknown"
        sub = best_dir / loc
        sub.mkdir(exist_ok=True)
        dest = _unique_dest(sub, p.name)
        shutil.move(str(p), str(dest))
        counts[loc] = counts.get(loc, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="AutoCull — automatically group and cull burst photos"
    )
    parser.add_argument("--input", required=True, help="Folder containing photos")
    parser.add_argument("--output", required=True, help="Folder for sorted results")
    parser.add_argument(
        "--gap",
        type=int,
        default=None,
        help="Session gap in seconds (default: auto-detect from photo distribution)",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Minimum sharpness score to keep a photo (default: 100)",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="copy or move files (default: copy)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview results without moving or copying any files",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Save per-photo decision log to output/autocull_log.csv",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for images in subdirectories recursively",
    )
    args = parser.parse_args()

    run(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        gap=args.gap,
        blur_threshold=args.blur_threshold,
        mode=args.mode,
        dry_run=args.dry_run,
        log=args.log,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
