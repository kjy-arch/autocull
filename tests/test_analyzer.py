from pathlib import Path

import cv2
import numpy as np
import pytest

from analyzer import analyze


def _checkerboard(path: Path) -> Path:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(0, 100, 5):
        for j in range(0, 100, 5):
            if (i // 5 + j // 5) % 2 == 0:
                img[i : i + 5, j : j + 5] = 255
    cv2.imwrite(str(path), img)
    return path


def _solid(path: Path) -> Path:
    cv2.imwrite(str(path), np.full((100, 100, 3), 128, dtype=np.uint8))
    return path


class TestAnalyze:
    def test_returns_expected_keys(self, tmp_path):
        result = analyze(_checkerboard(tmp_path / "img.jpg"))
        assert set(result.keys()) == {"blur_score", "has_face", "eyes_closed", "smile_score", "face_count"}

    def test_blur_score_is_float(self, tmp_path):
        assert isinstance(analyze(_checkerboard(tmp_path / "img.jpg"))["blur_score"], float)

    def test_sharp_image_scores_higher_than_blurry(self, tmp_path):
        sharp = analyze(_checkerboard(tmp_path / "sharp.jpg"))["blur_score"]
        blurry = analyze(_solid(tmp_path / "blurry.jpg"))["blur_score"]
        assert sharp > blurry

    def test_missing_file_returns_zero_score(self, tmp_path):
        assert analyze(tmp_path / "missing.jpg") == {
            "blur_score": 0.0,
            "has_face": False,
            "eyes_closed": False,
            "smile_score": 0.0,
            "face_count": 0,
        }

    def test_plain_image_has_no_face(self, tmp_path):
        result = analyze(_checkerboard(tmp_path / "img.jpg"))
        assert result["has_face"] is False
        assert result["eyes_closed"] is False
        assert result["smile_score"] == 0.0
