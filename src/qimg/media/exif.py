from __future__ import annotations

try:
    from PIL import ExifTags, Image
except Exception:  # pragma: no cover - optional dependency fallback
    ExifTags = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]


EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()} if ExifTags else {}
GPS_TAGS = ExifTags.GPSTAGS if ExifTags else {}


def _parse_gps(gps_info: dict) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in gps_info.items():
        name = GPS_TAGS.get(key, str(key))
        parsed[name] = value
    return parsed


def extract_exif(path: str) -> dict[str, object]:
    out: dict[str, object] = {}
    if Image is None:
        return out
    try:
        with Image.open(path) as img:
            exif = img.getexif()
    except Exception:
        return out

    if not exif:
        return out

    dt = exif.get(EXIF_TAGS.get("DateTimeOriginal")) or exif.get(EXIF_TAGS.get("DateTime"))
    model = exif.get(EXIF_TAGS.get("Model"))
    lens = exif.get(EXIF_TAGS.get("LensModel"))
    gps = exif.get(EXIF_TAGS.get("GPSInfo"))

    if dt:
        out["datetime"] = str(dt)
    if model:
        out["camera_model"] = str(model)
    if lens:
        out["lens"] = str(lens)
    if isinstance(gps, dict):
        out["gps"] = _parse_gps(gps)

    return out
