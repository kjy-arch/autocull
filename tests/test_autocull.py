from pathlib import Path
from unittest.mock import patch

import piexif
import pytest
from PIL import Image

from autocull import _best_filename, _unique_dest, _find_exact_duplicates, _find_perceptual_duplicates, pick_best, run


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
            Path("a.jpg"): {"blur_score": 50.0, "has_face": False, "eyes_closed": False, "smile_score": 0.0, "face_count": 0},
            Path("b.jpg"): {"blur_score": 200.0, "has_face": False, "eyes_closed": False, "smile_score": 0.0, "face_count": 0},
            Path("c.jpg"): {"blur_score": 100.0, "has_face": False, "eyes_closed": False, "smile_score": 0.0, "face_count": 0},
        }
        assert pick_best(paths, analyses) == Path("b.jpg")

    def test_prefers_open_eyes_over_sharper_closed(self):
        paths = [Path("a.jpg"), Path("b.jpg")]
        analyses = {
            Path("a.jpg"): {"blur_score": 500.0, "has_face": True, "eyes_closed": True, "smile_score": 0.0, "face_count": 1},
            Path("b.jpg"): {"blur_score": 100.0, "has_face": True, "eyes_closed": False, "smile_score": 0.0, "face_count": 1},
        }
        assert pick_best(paths, analyses) == Path("b.jpg")

    def test_falls_back_to_sharpest_when_all_closed(self):
        paths = [Path("a.jpg"), Path("b.jpg")]
        analyses = {
            Path("a.jpg"): {"blur_score": 200.0, "has_face": True, "eyes_closed": True, "smile_score": 0.0, "face_count": 1},
            Path("b.jpg"): {"blur_score": 100.0, "has_face": True, "eyes_closed": True, "smile_score": 0.0, "face_count": 1},
        }
        assert pick_best(paths, analyses) == Path("a.jpg")

    def test_prefers_smiling_over_sharper_non_smiling(self):
        paths = [Path("a.jpg"), Path("b.jpg")]
        analyses = {
            Path("a.jpg"): {"blur_score": 500.0, "has_face": True, "eyes_closed": False, "smile_score": 0.1, "face_count": 1},
            Path("b.jpg"): {"blur_score": 100.0, "has_face": True, "eyes_closed": False, "smile_score": 0.8, "face_count": 1},
        }
        assert pick_best(paths, analyses) == Path("b.jpg")

    def test_falls_back_to_sharpest_when_none_smiling(self):
        paths = [Path("a.jpg"), Path("b.jpg")]
        analyses = {
            Path("a.jpg"): {"blur_score": 200.0, "has_face": True, "eyes_closed": False, "smile_score": 0.1, "face_count": 1},
            Path("b.jpg"): {"blur_score": 100.0, "has_face": True, "eyes_closed": False, "smile_score": 0.1, "face_count": 1},
        }
        assert pick_best(paths, analyses) == Path("a.jpg")

    def test_single_photo(self):
        paths = [Path("a.jpg")]
        analyses = {Path("a.jpg"): {"blur_score": 150.0, "has_face": True, "eyes_closed": False, "smile_score": 0.5, "face_count": 1}}
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
    @pytest.fixture(autouse=True)
    def no_clip(self):
        with patch("grouper._embed", return_value=None):
            yield

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

    def test_single_image_always_kept_regardless_of_blur(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "a.jpg", "2024:01:01 10:00:00", sharp=False)
        with patch("autocull.get_gps", return_value=None):
            run(input_dir, tmp_path / "out", gap=15, blur_threshold=10000.0, mode="copy")
        assert (tmp_path / "out" / "best" / "20240101_unknown.jpg").exists()

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

    def test_dry_run_creates_no_files(self, tmp_path, capsys):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "a.jpg", "2024:01:01 10:00:00")
        with patch("autocull.get_gps", return_value=None):
            run(input_dir, tmp_path / "out", gap=15, blur_threshold=0.0, mode="copy", dry_run=True)
        assert not (tmp_path / "out").exists()

    def test_dry_run_prints_dry_run_marker(self, tmp_path, capsys):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "a.jpg", "2024:01:01 10:00:00")
        with patch("autocull.get_gps", return_value=None):
            run(input_dir, tmp_path / "out", gap=15, blur_threshold=0.0, mode="copy", dry_run=True)
        assert "DRY-RUN" in capsys.readouterr().out

    def test_log_creates_csv(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "a.jpg", "2024:01:01 10:00:00")
        with patch("autocull.get_gps", return_value=None):
            run(input_dir, tmp_path / "out", gap=15, blur_threshold=0.0, mode="copy", log=True)
        assert (tmp_path / "out" / "autocull_log.csv").exists()

    def test_log_has_correct_columns(self, tmp_path):
        import csv as _csv
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _make_jpeg(input_dir / "a.jpg", "2024:01:01 10:00:00")
        with patch("autocull.get_gps", return_value=None):
            run(input_dir, tmp_path / "out", gap=15, blur_threshold=0.0, mode="copy", log=True)
        with open(tmp_path / "out" / "autocull_log.csv", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            cols = reader.fieldnames
        assert "filename" in cols and "result" in cols and "reason" in cols

    def test_recursive_finds_images_in_subdirectory(self, tmp_path):
        input_dir = tmp_path / "in"
        sub = input_dir / "sub"
        sub.mkdir(parents=True)
        _make_jpeg(input_dir / "a.jpg", "2024:01:01 10:00:00")
        _make_jpeg(sub / "b.jpg", "2024:01:01 10:00:05")
        with patch("autocull.get_gps", return_value=None):
            run(input_dir, tmp_path / "out", gap=15, blur_threshold=0.0, mode="copy", recursive=True)
        best_files = list((tmp_path / "out" / "best").iterdir())
        assert len(best_files) == 1


class TestExactDuplicates:
    def test_identical_files_detected(self, tmp_path):
        p1 = tmp_path / "a.jpg"
        p2 = tmp_path / "b.jpg"
        p1.write_bytes(b"same content")
        p2.write_bytes(b"same content")
        unique, dupes = _find_exact_duplicates([p1, p2])
        assert len(unique) == 1
        assert len(dupes) == 1

    def test_different_files_all_unique(self, tmp_path):
        p1 = tmp_path / "a.jpg"
        p2 = tmp_path / "b.jpg"
        p1.write_bytes(b"content a")
        p2.write_bytes(b"content b")
        unique, dupes = _find_exact_duplicates([p1, p2])
        assert len(unique) == 2
        assert len(dupes) == 0

    def test_first_occurrence_kept(self, tmp_path):
        p1 = tmp_path / "a.jpg"
        p2 = tmp_path / "b.jpg"
        p1.write_bytes(b"same")
        p2.write_bytes(b"same")
        unique, dupes = _find_exact_duplicates([p1, p2])
        assert unique[0] == p1
        assert dupes[0] == p2


class TestPerceptualDuplicates:
    def _make_analyses(self, paths, blur_scores):
        return {
            p: {"blur_score": s, "has_face": False, "eyes_closed": False, "smile_score": 0.0, "face_count": 0}
            for p, s in zip(paths, blur_scores)
        }

    def test_identical_images_are_perceptual_dupes(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00", sharp=True)
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:01", sharp=True)
        analyses = self._make_analyses([p1, p2], [200.0, 200.0])
        unique, dupes = _find_perceptual_duplicates([p1, p2], analyses)
        assert len(unique) == 1
        assert len(dupes) == 1

    def test_sharpest_copy_is_kept(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00", sharp=True)
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:01", sharp=True)
        # p1 has lower blur_score — p2 should be kept
        analyses = self._make_analyses([p1, p2], [50.0, 200.0])
        unique, dupes = _find_perceptual_duplicates([p1, p2], analyses)
        assert unique[0] == p2
        assert dupes[0] == p1

    def test_visually_different_images_not_flagged(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00", sharp=True)
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:01", sharp=False)
        analyses = self._make_analyses([p1, p2], [200.0, 10.0])
        unique, dupes = _find_perceptual_duplicates([p1, p2], analyses)
        assert len(unique) == 2
        assert len(dupes) == 0
