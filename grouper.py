from pathlib import Path
from datetime import datetime
from PIL import Image

EXIF_DATETIME_ORIGINAL = 36867
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_EXIF_IFD = 0x8769
_FALLBACK_GAP = 300.0  # 5-minute fallback when distribution has no clear boundary


def get_timestamp(path: Path) -> datetime | None:
    try:
        img = Image.open(path)
        dt_str = img.getexif().get_ifd(_EXIF_IFD).get(EXIF_DATETIME_ORIGINAL)
        if dt_str:
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def find_images(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]


def _find_session_threshold(gaps: list[float]) -> float:
    """
    1D Otsu thresholding on the gap distribution.
    Finds the split point that maximises between-class variance.
    Returns _FALLBACK_GAP when no clear boundary exists (< 2 gaps or
    the two clusters are too similar).
    """
    if len(gaps) < 2:
        return _FALLBACK_GAP

    arr = sorted(gaps)
    n = len(arr)
    best_var, best_i = 0.0, None

    for i in range(1, n):
        w_l, w_r = i / n, (n - i) / n
        mu_l = sum(arr[:i]) / i
        mu_r = sum(arr[i:]) / (n - i)
        between_var = w_l * w_r * (mu_l - mu_r) ** 2
        if between_var > best_var:
            best_var, best_i = between_var, i

    if best_i is None:
        return _FALLBACK_GAP

    mu_l = sum(arr[:best_i]) / best_i
    mu_r = sum(arr[best_i:]) / (n - best_i)

    # Require between-session gaps to be at least 5× longer than within-session gaps
    if mu_l <= 0 or mu_r / mu_l < 5:
        return _FALLBACK_GAP

    return (arr[best_i - 1] + arr[best_i]) / 2


def group_by_time(paths: list[Path], gap_seconds: int | None = None) -> list[list[Path]]:
    """
    Groups photos by session.
    If gap_seconds is given, use it as a fixed threshold.
    Otherwise, auto-detect the session boundary from the gap distribution.
    """
    timestamped = [(get_timestamp(p), p) for p in paths]
    timestamped = [(ts, p) for ts, p in timestamped if ts is not None]
    timestamped.sort(key=lambda x: x[0])

    if not timestamped:
        return []
    if len(timestamped) == 1:
        return [[timestamped[0][1]]]

    gaps = [
        (timestamped[i][0] - timestamped[i - 1][0]).total_seconds()
        for i in range(1, len(timestamped))
    ]

    threshold = float(gap_seconds) if gap_seconds is not None else _find_session_threshold(gaps)

    groups: list[list[Path]] = [[timestamped[0][1]]]
    for i, gap in enumerate(gaps):
        if gap > threshold:
            groups.append([])
        groups[-1].append(timestamped[i + 1][1])

    return groups
