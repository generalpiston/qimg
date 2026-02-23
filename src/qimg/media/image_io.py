from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterator

try:
    from PIL import Image, UnidentifiedImageError
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None  # type: ignore[assignment]
    UnidentifiedImageError = Exception  # type: ignore[assignment]

from qimg.media.exif import extract_exif
from qimg.media.phash import compute_phash

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".heic",
}


@dataclass(slots=True)
class FileStat:
    abs_path: Path
    rel_path: str
    size: int
    mtime: float


@dataclass(slots=True)
class ImageMetadata:
    width: int | None
    height: int | None
    exif_json: str | None
    sha256: str | None
    phash: str | None


def _exts_from_mask(mask: str) -> set[str]:
    if "{" in mask and "}" in mask:
        brace = mask[mask.index("{") + 1 : mask.index("}")]
        items = [x.strip().lower() for x in brace.split(",") if x.strip()]
        return {f".{i.lstrip('.')}" for i in items}

    ext = Path(mask).suffix.lower()
    if ext:
        return {ext}
    return set(SUPPORTED_EXTENSIONS)


def iter_image_files(root: Path, mask: str) -> Iterator[FileStat]:
    exts = _exts_from_mask(mask)
    if not exts:
        exts = set(SUPPORTED_EXTENSIONS)

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in exts:
            continue
        rel_path = str(p.relative_to(root)).replace("\\", "/")
        st = p.stat()
        yield FileStat(abs_path=p.resolve(), rel_path=rel_path, size=st.st_size, mtime=st.st_mtime)


def _compute_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_metadata(path: Path, compute_hash: bool = False) -> ImageMetadata:
    width: int | None = None
    height: int | None = None

    if Image is None:
        return ImageMetadata(None, None, None, _compute_sha256(path) if compute_hash else None, None)

    try:
        with Image.open(path) as img:
            width, height = img.size
    except UnidentifiedImageError:
        return ImageMetadata(None, None, None, _compute_sha256(path) if compute_hash else None, None)
    except Exception:
        return ImageMetadata(None, None, None, _compute_sha256(path) if compute_hash else None, None)

    exif = extract_exif(str(path))
    exif_json = json.dumps(exif) if exif else None

    # HEIC might not be available in Pillow build; keep best-effort behavior.
    phash = compute_phash(str(path))
    return ImageMetadata(
        width=width,
        height=height,
        exif_json=exif_json,
        sha256=_compute_sha256(path) if compute_hash else None,
        phash=phash,
    )
