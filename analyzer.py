import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,
    refine_landmarks=True,
)

# MediaPipe Face Mesh eye landmark indices
_LEFT_EYE = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_CLOSED_THRESHOLD = 0.2


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
    img = cv2.imread(str(path))
    if img is None:
        return {"blur_score": 0.0, "has_face": False, "eyes_closed": False}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = _face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return {"blur_score": blur_score, "has_face": False, "eyes_closed": False}

    eyes_closed = any(
        _ear(face.landmark, _LEFT_EYE, w, h) < EAR_CLOSED_THRESHOLD
        or _ear(face.landmark, _RIGHT_EYE, w, h) < EAR_CLOSED_THRESHOLD
        for face in result.multi_face_landmarks
    )

    return {"blur_score": blur_score, "has_face": True, "eyes_closed": eyes_closed}
