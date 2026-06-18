import argparse
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


def run(input_dir: Path, output_dir: Path, gap: int | None, blur_threshold: float, mode: str):
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        return

    images = find_images(input_dir)
    if not images:
        print("No images found.")
        return

    print(f"Found {len(images)} images")
    workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(analyze, images),
            total=len(images),
            desc="Analyzing",
        ))
    all_analyses = dict(zip(images, results))
    face_counts = {p: all_analyses[p]["face_count"] for p in images}

    groups = group_by_time(images, gap_seconds=gap, use_clip=True, face_counts=face_counts)
    skipped = len(images) - sum(len(g) for g in groups)
    if skipped:
        print(f"Warning: {skipped} image(s) skipped — no EXIF timestamp")
    gap_info = f"{gap}s" if gap is not None else "auto"
    print(f"Grouped into {len(groups)} session(s) [gap={gap_info}]\n")

    best_dir = output_dir / "best"
    rejected_dir = output_dir / "rejected"
    best_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    transfer = shutil.copy2 if mode == "copy" else shutil.move

    kept = 0
    rejected = 0

    for i, group in enumerate(groups, 1):
        print(f"Session {i} ({len(group)} photo{'s' if len(group) > 1 else ''}):")
        analyses = {p: all_analyses[p] for p in group}

        if len(group) == 1:
            p = group[0]
            dest = _unique_dest(best_dir, _best_filename(p))
            transfer(str(p), str(dest))
            print(f"  [keep] {p.name} -> {dest.name}")
            kept += 1
            continue

        best = pick_best(group, analyses)

        for p in group:
            a = analyses[p]
            if p == best:
                dest = _unique_dest(best_dir, _best_filename(p))
                transfer(str(p), str(dest))
                print(f"  [keep] {p.name} -> {dest.name}")
                kept += 1
            else:
                if a["eyes_closed"]:
                    reason = "eyes closed"
                elif a["blur_score"] < blur_threshold:
                    reason = "blurry"
                else:
                    reason = "not best"
                transfer(str(p), str(rejected_dir / p.name))
                print(f"  [skip] {p.name} ({reason})")
                rejected += 1

    print(f"\nDone - kept: {kept}, rejected: {rejected}")
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
    args = parser.parse_args()

    run(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        gap=args.gap,
        blur_threshold=args.blur_threshold,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
