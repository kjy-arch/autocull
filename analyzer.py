import threading
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision
from pathlib import Path

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"

_LEFT_EYE = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_CLOSED_THRESHOLD = 0.2


def _ensure_model() -> None:
    if not _MODEL_PATH.exists():
        print("Downloading face landmark model (~23MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


_local = threading.local()

# Download the model once at import time so worker threads don't race on it
_ensure_model()


def _get_landmarker() -> _mp_vision.FaceLandmarker:
    if not hasattr(_local, "landmarker"):
        options = _mp_vision.FaceLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            num_faces=5,
            min_face_detection_confidence=0.5,
            output_face_blendshapes=True,
        )
        _local.landmarker = _mp_vision.FaceLandmarker.create_from_options(options)
    return _local.landmarker


_SMILE_NAMES = {"mouthSmileLeft", "mouthSmileRight"}
SMILE_THRESHOLD = 0.3

def _get_cascade() -> cv2.CascadeClassifier:
    if not hasattr(_local, "cascade"):
        _local.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _local.cascade


def _haar_face_count(img: np.ndarray) -> int:
    h, w = img.shape[:2]
    if w < 32 or h < 32:
        return 0
    scale = min(1.0, 1280 / w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return len(faces)


def _smile_score(blendshapes) -> float:
    """Average mouthSmile score across all detected faces."""
    if not blendshapes:
        return 0.0
    scores = []
    for face in blendshapes:
        vals = [c.score for c in face if c.category_name in _SMILE_NAMES]
        if vals:
            scores.append(sum(vals) / len(vals))
    return sum(scores) / len(scores) if scores else 0.0


def _ear(landmarks, indices: list[int], w: int, h: int) -> float:
    """Eye Aspect Ratio — below threshold means eye is closed."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    hz = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * hz) if hz > 0 else 0.0


def analyze(path: Path) -> dict:
    """
    Returns:
        blur_score  float  — higher = sharper (Laplacian variance)
        has_face    bool
        eyes_closed bool   — True if any detected face has a closed eye
    """
    # cv2.imread fails on non-ASCII paths on Windows; read via numpy instead
    try:
        buf = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        return {"blur_score": 0.0, "has_face": False, "eyes_closed": False, "smile_score": 0.0, "face_count": 0}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_landmarker().detect(mp_image)

    if not result.face_landmarks:
        haar_count = _haar_face_count(img)
        if haar_count > 0:
            # face_count=0 intentional: Haar counts differ from MediaPipe counts
            # so exclude from face_split to avoid spurious session breaks
            return {"blur_score": blur_score, "has_face": True, "eyes_closed": False, "smile_score": 0.0, "face_count": 0}
        return {"blur_score": blur_score, "has_face": False, "eyes_closed": False, "smile_score": 0.0, "face_count": 0}

    eyes_closed = any(
        _ear(face, _LEFT_EYE, w, h) < EAR_CLOSED_THRESHOLD
        or _ear(face, _RIGHT_EYE, w, h) < EAR_CLOSED_THRESHOLD
        for face in result.face_landmarks
    )

    return {
        "blur_score": blur_score,
        "has_face": True,
        "eyes_closed": eyes_closed,
        "smile_score": _smile_score(result.face_blendshapes),
        "face_count": len(result.face_landmarks),
    }
