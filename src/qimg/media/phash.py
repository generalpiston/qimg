from __future__ import annotations

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None  # type: ignore[assignment]

try:
    import imagehash
except Exception:  # pragma: no cover - optional dependency fallback
    imagehash = None  # type: ignore[assignment]


def compute_phash(path: str) -> str | None:
    if Image is None or imagehash is None:
        return None
    try:
        with Image.open(path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None


def hamming_distance_hex(a: str, b: str) -> int:
    if imagehash is None:
        raise RuntimeError("imagehash dependency is unavailable")
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
