from pathlib import Path
from unittest.mock import patch

import piexif
import pytest
from PIL import Image

from autocull import _best_filename, _unique_dest, pick_best, run


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


class TestBestFilename:
    def test_date_and_location(self, tmp_path):
        path = _make_jpeg(tmp_path / "a.jpg", "2024:06:15 10:30:00")
        with patch("autocull.get_gps", return_value=(37.5, 127.0)), \
             patch("autocull.place_name", return_value="강남구"):
            assert _best_filename(path) == "20240615_강남구.jpg"

    def test_falls_back_to_unknown_without_gps(self, tmp_path):
        path = _make_jpeg(tmp_path / "a.jpg", "2024:06:15 10:30:00")
        with patch("autocull.get_gps", return_value=None):
            assert _best_filename(path) == "20240615_unknown.jpg"

    def test_falls_back_when_geocoding_fails(self, tmp_path):
        path = _make_jpeg(tmp_path / "a.jpg", "2024:06:15 10:30:00")
        with patch("autocull.get_gps", return_value=(37.5, 127.0)), \
             patch("autocull.place_name", return_value=None):
            assert _best_filename(path) == "20240615_unknown.jpg"

    def test_preserves_suffix_lowercased(self, tmp_path):
        path = _make_jpeg(tmp_path / "A.JPG", "2024:06:15 10:30:00")
        with patch("autocull.get_gps", return_value=None):
            assert _best_filename(path).endswith(".jpg")


class TestUniqueDest:
    def test_returns_original_when_no_conflict(self, tmp_path):
        assert _unique_dest(tmp_path, "20240615_강남구.jpg") == tmp_path / "20240615_강남구.jpg"

    def test_appends_index_on_collision(self, tmp_path):
        (tmp_path / "20240615_강남구.jpg").write_bytes(b"")
        assert _unique_dest(tmp_path, "20240615_강남구.jpg") == tmp_path / "20240615_강남구_2.jpg"

    def test_increments_index_on_multiple_collisions(self, tmp_path):
        (tmp_path / "20240615_강남구.jpg").write_bytes(b"")
        (tmp_path / "20240615_강남구_2.jpg").write_bytes(b"")
        assert _unique_dest(tmp_path, "20240615_강남구.jpg") == tmp_path / "20240615_강남구_3.jpg"


class TestRun:
    def test_rejects_nonexistent_input(self, tmp_path, capsys):
        run(tmp_path / "nonexistent", tmp_path / "output", gap=15, blur_threshold=0.0, mode="copy")
        assert "Error" in capsys.readouterr().out

    def test_creates_output_directories(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "a.jpg", "2024:01:01 10:00:00")
        with patch("autocull.get_gps", return_value=None):
            run(input_dir, tmp_path / "out", gap=15, blur_threshold=0.0, mode="copy")
        assert (tmp_path / "out" / "best").exists()
        assert (tmp_path / "out" / "rejected").exists()

    def test_best_file_is_renamed_with_date_and_location(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "a.jpg", "2024:01:01 10:00:00", sharp=True)
        with patch("autocull.get_gps", return_value=(37.5, 127.0)), \
             patch("autocull.place_name", return_value="강남구"):
            run(input_dir, tmp_path / "out", gap=15, blur_threshold=0.0, mode="copy")
        assert (tmp_path / "out" / "best" / "20240101_강남구.jpg").exists()

    def test_best_file_falls_back_to_unknown_without_gps(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "a.jpg", "2024:01:01 10:00:00", sharp=True)
        with patch("autocull.get_gps", return_value=None):
            run(input_dir, tmp_path / "out", gap=15, blur_threshold=0.0, mode="copy")
        assert (tmp_path / "out" / "best" / "20240101_unknown.jpg").exists()

    def test_blurry_single_image_goes_to_rejected(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "a.jpg", "2024:01:01 10:00:00", sharp=False)
        with patch("autocull.get_gps", return_value=None):
            run(input_dir, tmp_path / "out", gap=15, blur_threshold=10000.0, mode="copy")
        assert (tmp_path / "out" / "rejected" / "a.jpg").exists()

    def test_best_of_two_goes_to_best_rejected_keeps_original_name(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "sharp.jpg", "2024:01:01 10:00:00", sharp=True)
        _make_jpeg(input_dir / "blurry.jpg", "2024:01:01 10:00:01", sharp=False)
        with patch("autocull.get_gps", return_value=None):
            run(input_dir, tmp_path / "out", gap=15, blur_threshold=0.0, mode="copy")
        best_files = list((tmp_path / "out" / "best").iterdir())
        assert len(best_files) == 1
        assert best_files[0].name.startswith("20240101_")
        assert (tmp_path / "out" / "rejected" / "blurry.jpg").exists()

    def test_warns_about_images_without_exif(self, tmp_path, capsys):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "with_exif.jpg", "2024:01:01 10:00:00")
        _make_jpeg(input_dir / "no_exif.jpg")
        with patch("autocull.get_gps", return_value=None):
            run(input_dir, tmp_path / "out", gap=15, blur_threshold=0.0, mode="copy")
        assert "Warning" in capsys.readouterr().out
