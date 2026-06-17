import re
import time
from pathlib import Path
from PIL import Image

_EXIF_GPS_IFD = 0x8825
_GPS_LAT_REF = 1
_GPS_LAT = 2
_GPS_LON_REF = 3
_GPS_LON = 4

_cache: dict[tuple[float, float], str | None] = {}
_last_request: float = 0.0


def get_gps(path: Path) -> tuple[float, float] | None:
    try:
        gps = Image.open(path).getexif().get_ifd(_EXIF_GPS_IFD)
        if not gps or _GPS_LAT not in gps:
            return None
        lat = _dms_to_decimal(gps[_GPS_LAT], gps.get(_GPS_LAT_REF, "N"))
        lon = _dms_to_decimal(gps[_GPS_LON], gps.get(_GPS_LON_REF, "E"))
        return lat, lon
    except Exception:
        return None


def _dms_to_decimal(dms, ref) -> float:
    d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
    decimal = d + m / 60 + s / 3600
    if isinstance(ref, bytes):
        ref = ref.decode()
    return -decimal if ref in ("S", "W") else decimal


def place_name(lat: float, lon: float) -> str | None:
    key = (round(lat, 3), round(lon, 3))
    if key not in _cache:
        _cache[key] = _reverse_geocode(lat, lon)
    return _cache[key]


def _reverse_geocode(lat: float, lon: float) -> str | None:
    global _last_request
    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        return None

    elapsed = time.time() - _last_request
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    try:
        geolocator = Nominatim(user_agent="autocull/1.0")
        result = geolocator.reverse((lat, lon), language="ko", timeout=5)
        _last_request = time.time()

        if not result:
            return None

        addr = result.raw.get("address", {})
        for key in ("suburb", "neighbourhood", "city_district", "county", "city"):
            name = addr.get(key)
            if name:
                return _sanitize(name)
    except Exception:
        pass
    return None


def _sanitize(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\s]', '', name)
