from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Collection:
    id: int
    name: str
    root_path: str
    mask: str
    created_at: str


@dataclass(slots=True)
class AssetRecord:
    id: str
    collection_id: int
    rel_path: str
    abs_path: str
    media_type: str
    mtime: float
    size: int
    width: int | None
    height: int | None
    sha256: str | None
    phash: str | None
    created_at: str
    updated_at: str
    is_deleted: int = 0
