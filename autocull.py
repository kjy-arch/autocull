import argparse
import shutil
from pathlib import Path

from grouper import find_images, group_by_time
from analyzer import analyze


def pick_best(group: list[Path], analyses: dict) -> Path:
    has_any_face = any(analyses[p]["has_face"] for p in group)

    candidates = group
    if has_any_face:
        open_eyes = [p for p in group if not analyses[p]["eyes_closed"]]
        if open_eyes:
            candidates = open_eyes

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

    groups = group_by_time(images, gap_seconds=gap)
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
        analyses = {p: analyze(p) for p in group}

        if len(group) == 1:
            p = group[0]
            if analyses[p]["blur_score"] >= blur_threshold:
                transfer(str(p), str(best_dir / p.name))
                print(f"  ✓ {p.name}")
                kept += 1
            else:
                transfer(str(p), str(rejected_dir / p.name))
                print(f"  ✗ {p.name} (blurry)")
                rejected += 1
            continue

        best = pick_best(group, analyses)

        for p in group:
            a = analyses[p]
            if p == best and a["blur_score"] >= blur_threshold:
                transfer(str(p), str(best_dir / p.name))
                print(f"  ✓ {p.name}")
                kept += 1
            else:
                if a["eyes_closed"]:
                    reason = "eyes closed"
                elif a["blur_score"] < blur_threshold:
                    reason = "blurry"
                else:
                    reason = "not best"
                transfer(str(p), str(rejected_dir / p.name))
                print(f"  ✗ {p.name} ({reason})")
                rejected += 1

    print(f"\nDone — kept: {kept}, rejected: {rejected}")
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
