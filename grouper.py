import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm

EXIF_DATETIME_ORIGINAL = 36867
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".wmv", ".mts", ".ts"}
_EXIF_IFD = 0x8769
_FALLBACK_GAP = 120.0  # 2-minute fallback when distribution has no clear boundary
_SCENE_SIM_THRESHOLD = 0.75  # cosine similarity below this → new session

_clip_model = None
_clip_preprocess = None
_clip_lock = threading.Lock()


def _get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        with _clip_lock:
            if _clip_model is None:
                import open_clip
                print("Loading CLIP model...")
                _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai"
                )
                _clip_model.eval()
    return _clip_model, _clip_preprocess


def _embed(path: Path):
    try:
        import torch
        model, preprocess = _get_clip()
        img = preprocess(Image.open(path)).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb
    except Exception:
        return None


def _cosine_sim(a, b) -> float:
    if a is None or b is None:
        return 1.0
    try:
        return float((a * b).sum())
    except Exception:
        return 1.0


def get_timestamp(path: Path) -> datetime | None:
    try:
        img = Image.open(path)
        dt_str = img.getexif().get_ifd(_EXIF_IFD).get(EXIF_DATETIME_ORIGINAL)
        if dt_str:
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def find_images(folder: Path, recursive: bool = False, exclude: Path | list[Path] | None = None) -> list[Path]:
    paths = folder.rglob("*") if recursive else folder.iterdir()
    excludes = [exclude] if isinstance(exclude, Path) else (exclude or [])
    return [
        p for p in paths
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(p.is_relative_to(e) for e in excludes)
    ]


def find_videos(folder: Path, recursive: bool = False, exclude: Path | list[Path] | None = None) -> list[Path]:
    paths = folder.rglob("*") if recursive else folder.iterdir()
    excludes = [exclude] if isinstance(exclude, Path) else (exclude or [])
    return [
        p for p in paths
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and not any(p.is_relative_to(e) for e in excludes)
    ]


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

    # Cap at FALLBACK_GAP so years-spanning libraries get sensible splits
    return min((arr[best_i - 1] + arr[best_i]) / 2, _FALLBACK_GAP)


def _face_count_changed(face_counts: dict | None, p1: Path, p2: Path) -> bool:
    if not face_counts:
        return False
    c1 = face_counts.get(p1, 0)
    c2 = face_counts.get(p2, 0)
    # 둘 다 얼굴이 감지됐을 때만 비교 (한쪽이 0이면 감지 실패로 간주)
    if c1 == 0 or c2 == 0:
        return False
    return c1 != c2


def group_by_time(
    paths: list[Path],
    gap_seconds: int | None = None,
    use_clip: bool = False,
    face_counts: dict | None = None,
) -> list[list[Path]]:
    """
    Groups photos by session.
    If gap_seconds is given, use it as a fixed time threshold.
    Otherwise, auto-detect from gap distribution.
    If use_clip is True, also split when scene similarity drops below _SCENE_SIM_THRESHOLD.
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

    embeddings = None
    if use_clip:
        workers = min(4, os.cpu_count() or 1)
        paths_only = [p for _, p in timestamped]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            embeddings = list(tqdm(
                executor.map(_embed, paths_only),
                total=len(paths_only),
                desc="CLIP embeddings",
            ))

    groups: list[list[Path]] = [[timestamped[0][1]]]
    for i, gap in enumerate(gaps):
        p_cur = timestamped[i][1]
        p_next = timestamped[i + 1][1]
        time_split = gap > threshold
        scene_split = (
            embeddings is not None
            and _cosine_sim(embeddings[i], embeddings[i + 1]) < _SCENE_SIM_THRESHOLD
        )
        face_split = _face_count_changed(face_counts, p_cur, p_next)
        if time_split or scene_split or face_split:
            groups.append([])
        groups[-1].append(p_next)

    return groups
