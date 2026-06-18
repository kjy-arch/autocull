from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import piexif
import pytest
from PIL import Image

from grouper import _FALLBACK_GAP, _find_session_threshold, find_images, get_timestamp, group_by_time


def _make_jpeg(path: Path, dt_str: str | None = None) -> Path:
    img = Image.new("RGB", (10, 10), (128, 128, 128))
    if dt_str:
        exif_bytes = piexif.dump({"Exif": {piexif.ExifIFD.DateTimeOriginal: dt_str.encode()}})
        img.save(str(path), "JPEG", exif=exif_bytes)
    else:
        img.save(str(path), "JPEG")
    return path


class TestGetTimestamp:
    def test_returns_datetime_for_jpeg_with_exif(self, tmp_path):
        path = _make_jpeg(tmp_path / "a.jpg", "2024:06:15 10:30:00")
        assert get_timestamp(path) == datetime(2024, 6, 15, 10, 30, 0)

    def test_returns_none_without_exif(self, tmp_path):
        path = _make_jpeg(tmp_path / "a.jpg")
        assert get_timestamp(path) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        assert get_timestamp(tmp_path / "missing.jpg") is None


class TestGroupByTime:
    def test_empty_returns_empty(self):
        assert group_by_time([]) == []

    def test_single_image_forms_one_group(self, tmp_path):
        p = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        assert group_by_time([p], gap_seconds=15) == [[p]]

    def test_two_images_within_gap_are_grouped(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:10")
        groups = group_by_time([p1, p2], gap_seconds=15)
        assert len(groups) == 1
        assert set(groups[0]) == {p1, p2}

    def test_two_images_beyond_gap_form_separate_groups(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:01:00")
        assert len(group_by_time([p1, p2], gap_seconds=15)) == 2

    def test_images_without_timestamp_are_excluded(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg")
        groups = group_by_time([p1, p2], gap_seconds=15)
        assert len(groups) == 1
        assert groups[0] == [p1]

    def test_groups_are_sorted_by_timestamp(self, tmp_path):
        p_late = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:10")
        p_early = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:00")
        groups = group_by_time([p_late, p_early], gap_seconds=15)
        assert groups[0] == [p_early, p_late]


class TestFindImages:
    def test_finds_jpeg_files(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"")
        (tmp_path / "b.jpeg").write_bytes(b"")
        (tmp_path / "c.JPG").write_bytes(b"")
        assert len(find_images(tmp_path)) == 3

    def test_ignores_non_image_files(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"")
        (tmp_path / "b.txt").write_bytes(b"")
        (tmp_path / "c.mp4").write_bytes(b"")
        assert len(find_images(tmp_path)) == 1

    def test_recursive_finds_files_in_subdirectory(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.jpg").write_bytes(b"")
        (sub / "b.jpg").write_bytes(b"")
        assert len(find_images(tmp_path, recursive=True)) == 2

    def test_not_recursive_ignores_subdirectory(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.jpg").write_bytes(b"")
        (sub / "b.jpg").write_bytes(b"")
        assert len(find_images(tmp_path, recursive=False)) == 1

    def test_exclude_skips_matching_subtree(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        (tmp_path / "a.jpg").write_bytes(b"")
        (out / "b.jpg").write_bytes(b"")
        assert len(find_images(tmp_path, recursive=True, exclude=out)) == 1


class TestFindSessionThreshold:
    def test_clear_boundary_returns_midpoint(self):
        # 10 short gaps + 1 large gap → clear boundary
        gaps = [1.0] * 10 + [300.0]
        t = _find_session_threshold(gaps)
        assert 1.0 < t < 300.0

    def test_uniform_gaps_return_fallback(self):
        # No clear boundary → fallback
        assert _find_session_threshold([1.0] * 20) == _FALLBACK_GAP

    def test_single_gap_returns_fallback(self):
        assert _find_session_threshold([5.0]) == _FALLBACK_GAP

    def test_two_clear_clusters(self):
        gaps = [1.0] * 5 + [600.0] * 2
        t = _find_session_threshold(gaps)
        assert 1.0 < t < 600.0

    def test_small_ratio_returns_fallback(self):
        # 1s vs 2s is not a meaningful session boundary
        gaps = [1.0] * 10 + [2.0] * 10
        assert _find_session_threshold(gaps) == _FALLBACK_GAP


class TestAdaptiveGrouping:
    def test_single_burst_stays_together(self, tmp_path):
        photos = [_make_jpeg(tmp_path / f"p{i}.jpg", f"2024:01:01 10:00:0{i}") for i in range(5)]
        with patch("grouper._embed", return_value=None):
            assert len(group_by_time(photos)) == 1

    def test_two_bursts_separated_by_minutes_split_automatically(self, tmp_path):
        burst1 = [_make_jpeg(tmp_path / f"a{i}.jpg", f"2024:01:01 10:00:0{i}") for i in range(5)]
        burst2 = [_make_jpeg(tmp_path / f"b{i}.jpg", f"2024:01:01 10:10:0{i}") for i in range(5)]
        with patch("grouper._embed", return_value=None):
            groups = group_by_time(burst1 + burst2)
        assert len(groups) == 2

    def test_gap_seconds_overrides_auto_detection(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:01:00")
        assert len(group_by_time([p1, p2], gap_seconds=15)) == 2
        with patch("grouper._embed", return_value=None):
            assert len(group_by_time([p1, p2])) == 1

    def test_clip_splits_scene_change_within_time_threshold(self, tmp_path):
        import torch
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:10")
        # Simulate very different embeddings (cosine sim ≈ 0)
        emb_a = torch.tensor([[1.0, 0.0]])
        emb_b = torch.tensor([[0.0, 1.0]])
        with patch("grouper._embed", side_effect=[emb_a, emb_b]):
            groups = group_by_time([p1, p2], gap_seconds=30, use_clip=True)
        assert len(groups) == 2

    def test_clip_keeps_similar_scenes_together(self, tmp_path):
        import torch
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:10")
        # Simulate identical embeddings (cosine sim = 1.0)
        emb = torch.tensor([[1.0, 0.0]])
        with patch("grouper._embed", return_value=emb):
            groups = group_by_time([p1, p2], gap_seconds=30, use_clip=True)
        assert len(groups) == 1

    def test_face_count_change_splits_session(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:05")
        face_counts = {p1: 2, p2: 1}
        groups = group_by_time([p1, p2], gap_seconds=30, face_counts=face_counts)
        assert len(groups) == 2

    def test_face_count_same_stays_together(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:05")
        face_counts = {p1: 2, p2: 2}
        groups = group_by_time([p1, p2], gap_seconds=30, face_counts=face_counts)
        assert len(groups) == 1

    def test_zero_face_count_not_split(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:05")
        face_counts = {p1: 0, p2: 2}
        groups = group_by_time([p1, p2], gap_seconds=30, face_counts=face_counts)
        assert len(groups) == 1
