from pathlib import Path
from unittest.mock import patch

import piexif
import pytest
from PIL import Image

import location
from location import _sanitize, get_gps, place_name


def _make_jpeg_with_gps(path: Path, lat: float, lon: float) -> Path:
    def to_dms(value: float):
        d = int(abs(value))
        m_float = (abs(value) - d) * 60
        m = int(m_float)
        s = (m_float - m) * 60
        return ((d, 1), (m, 1), (int(s * 1000), 1000))

    img = Image.new("RGB", (10, 10))
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
        piexif.GPSIFD.GPSLatitude: to_dms(lat),
        piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
        piexif.GPSIFD.GPSLongitude: to_dms(lon),
    }
    img.save(str(path), "JPEG", exif=piexif.dump({"GPS": gps_ifd}))
    return path


@pytest.fixture(autouse=True)
def clear_cache():
    location._cache.clear()
    yield
    location._cache.clear()


class TestGetGps:
    def test_returns_coords_from_gps_jpeg(self, tmp_path):
        path = _make_jpeg_with_gps(tmp_path / "a.jpg", 37.5, 127.0)
        coords = get_gps(path)
        assert coords is not None
        lat, lon = coords
        assert abs(lat - 37.5) < 0.01
        assert abs(lon - 127.0) < 0.01

    def test_returns_none_without_gps(self, tmp_path):
        path = tmp_path / "a.jpg"
        Image.new("RGB", (10, 10)).save(str(path), "JPEG")
        assert get_gps(path) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        assert get_gps(tmp_path / "missing.jpg") is None

    def test_southern_hemisphere_is_negative(self, tmp_path):
        path = _make_jpeg_with_gps(tmp_path / "a.jpg", -33.9, 151.2)
        coords = get_gps(path)
        assert coords is not None
        assert coords[0] < 0  # latitude is negative (south)


class TestPlaceName:
    def test_returns_place_from_reverse_geocode(self):
        with patch("location._reverse_geocode", return_value="강남구"):
            assert place_name(37.5, 127.0) == "강남구"

    def test_returns_none_when_geocode_fails(self):
        with patch("location._reverse_geocode", return_value=None):
            assert place_name(0.0, 0.0) is None

    def test_caches_repeated_calls_at_same_coords(self):
        with patch("location._reverse_geocode", return_value="강남구") as mock:
            place_name(37.517, 127.047)
            place_name(37.517, 127.047)
        assert mock.call_count == 1

    def test_nearby_coords_share_cache(self):
        # Coordinates differing by less than 0.001° share the same cache slot
        with patch("location._reverse_geocode", return_value="역삼동") as mock:
            place_name(37.5001, 127.0001)
            place_name(37.5002, 127.0002)
        assert mock.call_count == 1


class TestSanitize:
    def test_removes_path_separators(self):
        assert "/" not in _sanitize("some/path")
        assert "\\" not in _sanitize("back\\slash")

    def test_removes_whitespace(self):
        assert " " not in _sanitize("강남 구")

    def test_preserves_korean(self):
        assert _sanitize("강남구") == "강남구"
