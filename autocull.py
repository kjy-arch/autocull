import argparse
import csv
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from grouper import find_images, group_by_time, get_timestamp
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


def _find_duplicates(images: list[Path]) -> tuple[list[Path], list[Path]]:
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

    images = find_images(input_dir, recursive=recursive, exclude=output_dir)
    if not images:
        print("No images found.")
        return

    print(f"Found {len(images)} images")

    unique, dupes = _find_duplicates(images)
    if dupes:
        print(f"Found {len(dupes)} duplicate(s) — will be rejected")

    workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(analyze, unique),
            total=len(unique),
            desc="Analyzing",
        ))
    all_analyses = dict(zip(unique, results))
    face_counts = {p: all_analyses[p]["face_count"] for p in unique}

    groups = group_by_time(unique, gap_seconds=gap, use_clip=True, face_counts=face_counts)
    skipped = len(unique) - sum(len(g) for g in groups)
    if skipped:
        print(f"Warning: {skipped} image(s) skipped — no EXIF timestamp")
    gap_info = f"{gap}s" if gap is not None else "auto"
    print(f"Grouped into {len(groups)} session(s) [gap={gap_info}]")

    if dry_run:
        print("[DRY-RUN] No files will be moved or copied.\n")
    else:
        print()
        best_dir = output_dir / "best"
        rejected_dir = output_dir / "rejected"
        best_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir.mkdir(parents=True, exist_ok=True)

    transfer = shutil.copy2 if mode == "copy" else shutil.move

    kept = 0
    rejected = 0
    log_rows: list[dict] = []

    def _log(session, path, result, reason, analysis, dest=""):
        if not log:
            return
        row = {"session": session, "filename": path.name, "result": result, "reason": reason, "dest": dest}
        if analysis:
            row.update({k: analysis[k] for k in ("blur_score", "has_face", "eyes_closed", "smile_score", "face_count")})
        else:
            row.update({"blur_score": "", "has_face": "", "eyes_closed": "", "smile_score": "", "face_count": ""})
        log_rows.append(row)

    for p in dupes:
        if not dry_run:
            transfer(str(p), str(rejected_dir / p.name))
        print(f"  [skip] {p.name} (duplicate)")
        rejected += 1
        _log("", p, "skip", "duplicate", None)

    for i, group in enumerate(groups, 1):
        print(f"Session {i} ({len(group)} photo{'s' if len(group) > 1 else ''}):")
        analyses = {p: all_analyses[p] for p in group}

        if len(group) == 1:
            p = group[0]
            dest_name = _best_filename(p)
            if not dry_run:
                dest = _unique_dest(best_dir, dest_name)
                transfer(str(p), str(dest))
                dest_name = dest.name
            print(f"  [keep] {p.name} -> {dest_name}")
            kept += 1
            _log(i, p, "keep", "", analyses[p], dest_name)
            continue

        best = pick_best(group, analyses)

        for p in group:
            a = analyses[p]
            if p == best:
                dest_name = _best_filename(p)
                if not dry_run:
                    dest = _unique_dest(best_dir, dest_name)
                    transfer(str(p), str(dest))
                    dest_name = dest.name
                print(f"  [keep] {p.name} -> {dest_name}")
                kept += 1
                _log(i, p, "keep", "", a, dest_name)
            else:
                if a["eyes_closed"]:
                    reason = "eyes closed"
                elif a["blur_score"] < blur_threshold:
                    reason = "blurry"
                else:
                    reason = "not best"
                if not dry_run:
                    transfer(str(p), str(rejected_dir / p.name))
                print(f"  [skip] {p.name} ({reason})")
                rejected += 1
                _log(i, p, "skip", reason, a)

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
