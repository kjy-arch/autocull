from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import piexif
import pytest
from PIL import Image

from datetime import datetime
from grouper import _FALLBACK_GAP, _find_session_threshold, find_images, get_timestamp, group_by_time, split_by_clip, has_exif_timestamp, cluster_by_clip


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

    def test_falls_back_to_mtime_without_exif(self, tmp_path):
        path = _make_jpeg(tmp_path / "a.jpg")  # 비숫자 파일명, EXIF 없음
        assert isinstance(get_timestamp(path), datetime)

    def test_parses_13digit_unix_ms_timestamp_from_filename(self, tmp_path):
        path = _make_jpeg(tmp_path / "1379498137130.jpg")
        assert get_timestamp(path) == datetime.fromtimestamp(1379498137)

    def test_parses_10digit_unix_seconds_timestamp_from_filename(self, tmp_path):
        path = _make_jpeg(tmp_path / "1379498137.jpg")
        assert get_timestamp(path) == datetime.fromtimestamp(1379498137)

    def test_exif_takes_priority_over_filename_timestamp(self, tmp_path):
        path = _make_jpeg(tmp_path / "1379498137130.jpg", "2024:06:15 10:30:00")
        assert get_timestamp(path) == datetime(2024, 6, 15, 10, 30, 0)

    def test_returns_none_for_missing_file(self, tmp_path):
        assert get_timestamp(tmp_path / "missing.jpg") is None


class TestGroupByTime:
    def test_empty_returns_empty(self):
        groups, _ = group_by_time([])
        assert groups == []

    def test_single_image_forms_one_group(self, tmp_path):
        p = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        assert group_by_time([p], gap_seconds=15)[0] == [[p]]

    def test_two_images_within_gap_are_grouped(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:10")
        groups, _ = group_by_time([p1, p2], gap_seconds=15)
        assert len(groups) == 1
        assert set(groups[0]) == {p1, p2}

    def test_two_images_beyond_gap_form_separate_groups(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:01:00")
        assert len(group_by_time([p1, p2], gap_seconds=15)[0]) == 2

    def test_images_without_exif_included_via_mtime(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg")  # EXIF 없음 → mtime fallback
        groups, _ = group_by_time([p1, p2], gap_seconds=15)
        all_photos = [p for g in groups for p in g]
        assert p1 in all_photos
        assert p2 in all_photos

    def test_groups_are_sorted_by_timestamp(self, tmp_path):
        p_late = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:10")
        p_early = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:00")
        groups, _ = group_by_time([p_late, p_early], gap_seconds=15)
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
            assert len(group_by_time(photos)[0]) == 1

    def test_two_bursts_separated_by_minutes_split_automatically(self, tmp_path):
        burst1 = [_make_jpeg(tmp_path / f"a{i}.jpg", f"2024:01:01 10:00:0{i}") for i in range(5)]
        burst2 = [_make_jpeg(tmp_path / f"b{i}.jpg", f"2024:01:01 10:10:0{i}") for i in range(5)]
        with patch("grouper._embed", return_value=None):
            groups, _ = group_by_time(burst1 + burst2)
        assert len(groups) == 2

    def test_gap_seconds_overrides_auto_detection(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:01:00")
        assert len(group_by_time([p1, p2], gap_seconds=15)[0]) == 2
        with patch("grouper._embed", return_value=None):
            assert len(group_by_time([p1, p2])[0]) == 1

    def test_clip_splits_scene_change_within_time_threshold(self, tmp_path):
        import torch
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:10")
        # Simulate very different embeddings (cosine sim ≈ 0)
        emb_a = torch.tensor([[1.0, 0.0]])
        emb_b = torch.tensor([[0.0, 1.0]])
        with patch("grouper._embed", side_effect=[emb_a, emb_b]):
            groups, _ = group_by_time([p1, p2], gap_seconds=30, use_clip=True)
        assert len(groups) == 2

    def test_clip_keeps_similar_scenes_together(self, tmp_path):
        import torch
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:10")
        # Simulate identical embeddings (cosine sim = 1.0)
        emb = torch.tensor([[1.0, 0.0]])
        with patch("grouper._embed", return_value=emb):
            groups, _ = group_by_time([p1, p2], gap_seconds=30, use_clip=True)
        assert len(groups) == 1

    def test_face_count_change_splits_session(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:05")
        face_counts = {p1: 2, p2: 1}
        groups, _ = group_by_time([p1, p2], gap_seconds=30, face_counts=face_counts)
        assert len(groups) == 2

    def test_face_count_same_stays_together(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:05")
        face_counts = {p1: 2, p2: 2}
        groups, _ = group_by_time([p1, p2], gap_seconds=30, face_counts=face_counts)
        assert len(groups) == 1

    def test_zero_face_count_not_split(self, tmp_path):
        p1 = _make_jpeg(tmp_path / "a.jpg", "2024:01:01 10:00:00")
        p2 = _make_jpeg(tmp_path / "b.jpg", "2024:01:01 10:00:05")
        face_counts = {p1: 0, p2: 2}
        groups, _ = group_by_time([p1, p2], gap_seconds=30, face_counts=face_counts)
        assert len(groups) == 1


class TestHasExifTimestamp:
    def test_returns_true_with_exif(self, tmp_path):
        path = _make_jpeg(tmp_path / "a.jpg", "2024:06:15 10:30:00")
        assert has_exif_timestamp(path) is True

    def test_returns_false_without_exif(self, tmp_path):
        path = _make_jpeg(tmp_path / "a.jpg")
        assert has_exif_timestamp(path) is False

    def test_returns_false_for_missing_file(self, tmp_path):
        assert has_exif_timestamp(tmp_path / "missing.jpg") is False


class TestClusterByClip:
    def test_empty_returns_empty(self):
        assert cluster_by_clip([]) == ([], {})

    def test_single_photo_forms_one_cluster(self, tmp_path):
        p = _make_jpeg(tmp_path / "a.jpg")
        with patch("grouper._embed", return_value=None):
            clusters, _ = cluster_by_clip([p])
        assert clusters == [[p]]

    def test_similar_photos_stay_together(self, tmp_path):
        import torch
        p1, p2 = _make_jpeg(tmp_path / "a.jpg"), _make_jpeg(tmp_path / "b.jpg")
        emb = torch.tensor([[1.0, 0.0]])
        with patch("grouper._embed", return_value=emb):
            clusters, _ = cluster_by_clip([p1, p2])
        assert len(clusters) == 1

    def test_dissimilar_photos_split(self, tmp_path):
        import torch
        p1, p2 = _make_jpeg(tmp_path / "a.jpg"), _make_jpeg(tmp_path / "b.jpg")
        embs = iter([torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])])
        with patch("grouper._embed", side_effect=lambda _: next(embs)):
            clusters, _ = cluster_by_clip([p1, p2])
        assert len(clusters) == 2


class TestSplitByClip:
    def test_single_photo_returns_one_cluster(self, tmp_path):
        p = tmp_path / "a.jpg"
        p.write_bytes(b"")
        assert split_by_clip([p], {}) == [[p]]

    def test_empty_embeddings_groups_all_together(self, tmp_path):
        p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
        p1.write_bytes(b"")
        p2.write_bytes(b"")
        assert len(split_by_clip([p1, p2], {})) == 1

    def test_similar_embeddings_stay_in_one_cluster(self, tmp_path):
        import torch
        p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
        p1.write_bytes(b"")
        p2.write_bytes(b"")
        emb = torch.tensor([[1.0, 0.0]])
        clusters = split_by_clip([p1, p2], {p1: emb, p2: emb})
        assert len(clusters) == 1
        assert set(clusters[0]) == {p1, p2}

    def test_dissimilar_embeddings_split_into_two_clusters(self, tmp_path):
        import torch
        p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
        p1.write_bytes(b"")
        p2.write_bytes(b"")
        emb1 = torch.tensor([[1.0, 0.0]])
        emb2 = torch.tensor([[0.0, 1.0]])  # cosine sim = 0.0 < threshold
        clusters = split_by_clip([p1, p2], {p1: emb1, p2: emb2})
        assert len(clusters) == 2

    def test_three_photos_two_scenes(self, tmp_path):
        import torch
        p1, p2, p3 = [tmp_path / f"{n}.jpg" for n in "abc"]
        for p in (p1, p2, p3):
            p.write_bytes(b"")
        emb_a = torch.tensor([[1.0, 0.0]])
        emb_b = torch.tensor([[0.0, 1.0]])
        # p1 and p2 are scene A, p3 is scene B
        clusters = split_by_clip([p1, p2, p3], {p1: emb_a, p2: emb_a, p3: emb_b})
        assert len(clusters) == 2
        assert p3 in clusters[1]
