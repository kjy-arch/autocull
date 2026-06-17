from pathlib import Path
from datetime import datetime
from PIL import Image

EXIF_DATETIME_ORIGINAL = 36867
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_EXIF_IFD = 0x8769  # ExifIFD sub-directory pointer


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


def group_by_time(paths: list[Path], gap_seconds: int = 15) -> list[list[Path]]:
    timestamped = [(get_timestamp(p), p) for p in paths]
    timestamped = [(ts, p) for ts, p in timestamped if ts is not None]
    timestamped.sort(key=lambda x: x[0])

    if not timestamped:
        return []

    groups: list[list[Path]] = [[timestamped[0][1]]]
    for i in range(1, len(timestamped)):
        delta = (timestamped[i][0] - timestamped[i - 1][0]).total_seconds()
        if delta <= gap_seconds:
            groups[-1].append(timestamped[i][1])
        else:
            groups.append([timestamped[i][1]])

    return groups
