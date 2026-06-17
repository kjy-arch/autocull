from pathlib import Path

import piexif
import pytest
from PIL import Image

from autocull import pick_best, run


def _make_jpeg(path: Path, dt_str: str | None = None, sharp: bool = True) -> Path:
    if sharp:
        data = [255 if (i // 5 + j // 5) % 2 == 0 else 0 for i in range(100) for j in range(100)]
        img = Image.new("RGB", (100, 100))
        img.putdata([(v, v, v) for v in data])
    else:
        img = Image.new("RGB", (100, 100), (128, 128, 128))
    if dt_str:
        exif_bytes = piexif.dump({"Exif": {piexif.ExifIFD.DateTimeOriginal: dt_str.encode()}})
        img.save(str(path), "JPEG", exif=exif_bytes)
    else:
        img.save(str(path), "JPEG")
    return path


class TestPickBest:
    def test_picks_sharpest_when_no_faces(self):
        paths = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
        analyses = {
            Path("a.jpg"): {"blur_score": 50.0, "has_face": False, "eyes_closed": False},
            Path("b.jpg"): {"blur_score": 200.0, "has_face": False, "eyes_closed": False},
            Path("c.jpg"): {"blur_score": 100.0, "has_face": False, "eyes_closed": False},
        }
        assert pick_best(paths, analyses) == Path("b.jpg")

    def test_prefers_open_eyes_over_sharper_closed(self):
        paths = [Path("a.jpg"), Path("b.jpg")]
        analyses = {
            Path("a.jpg"): {"blur_score": 500.0, "has_face": True, "eyes_closed": True},
            Path("b.jpg"): {"blur_score": 100.0, "has_face": True, "eyes_closed": False},
        }
        assert pick_best(paths, analyses) == Path("b.jpg")

    def test_falls_back_to_sharpest_when_all_closed(self):
        paths = [Path("a.jpg"), Path("b.jpg")]
        analyses = {
            Path("a.jpg"): {"blur_score": 200.0, "has_face": True, "eyes_closed": True},
            Path("b.jpg"): {"blur_score": 100.0, "has_face": True, "eyes_closed": True},
        }
        assert pick_best(paths, analyses) == Path("a.jpg")

    def test_single_photo(self):
        paths = [Path("a.jpg")]
        analyses = {Path("a.jpg"): {"blur_score": 150.0, "has_face": True, "eyes_closed": False}}
        assert pick_best(paths, analyses) == Path("a.jpg")


class TestRun:
    def test_rejects_nonexistent_input(self, tmp_path, capsys):
        run(tmp_path / "nonexistent", tmp_path / "output", gap=15, blur_threshold=0.0, mode="copy")
        assert "Error" in capsys.readouterr().out

    def test_creates_output_directories(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        output_dir = tmp_path / "out"
        _make_jpeg(input_dir / "a.jpg", "2024:01:01 10:00:00")
        run(input_dir, output_dir, gap=15, blur_threshold=0.0, mode="copy")
        assert (output_dir / "best").exists()
        assert (output_dir / "rejected").exists()

    def test_single_sharp_image_goes_to_best(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "a.jpg", "2024:01:01 10:00:00", sharp=True)
        run(input_dir, tmp_path / "out", gap=15, blur_threshold=0.0, mode="copy")
        assert (tmp_path / "out" / "best" / "a.jpg").exists()

    def test_blurry_single_image_goes_to_rejected(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "a.jpg", "2024:01:01 10:00:00", sharp=False)
        run(input_dir, tmp_path / "out", gap=15, blur_threshold=10000.0, mode="copy")
        assert (tmp_path / "out" / "rejected" / "a.jpg").exists()

    def test_best_of_two_goes_to_best(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "sharp.jpg", "2024:01:01 10:00:00", sharp=True)
        _make_jpeg(input_dir / "blurry.jpg", "2024:01:01 10:00:01", sharp=False)
        run(input_dir, tmp_path / "out", gap=15, blur_threshold=0.0, mode="copy")
        assert (tmp_path / "out" / "best" / "sharp.jpg").exists()
        assert (tmp_path / "out" / "rejected" / "blurry.jpg").exists()

    def test_warns_about_images_without_exif(self, tmp_path, capsys):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "with_exif.jpg", "2024:01:01 10:00:00")
        _make_jpeg(input_dir / "no_exif.jpg")
        run(input_dir, tmp_path / "out", gap=15, blur_threshold=0.0, mode="copy")
        assert "Warning" in capsys.readouterr().out
