from __future__ import annotations

import hashlib


def short_id_from_int(value: int) -> str:
    return f"#{value:06x}"


def parse_short_id(value: str) -> int | None:
    raw = value.strip()
    if not raw.startswith("#"):
        return None
    text = raw[1:]
    if not text:
        return None
    try:
        return int(text, 16)
    except ValueError:
        return None


def normalize_asset_id(value: str) -> str:
    num = parse_short_id(value)
    if num is None:
        raise ValueError(f"invalid asset id: {value}")
    return short_id_from_int(num)


def stable_asset_id(collection_id: int, rel_path: str, collision_salt: int = 0) -> str:
    seed = f"{collection_id}:{rel_path.lower()}:{collision_salt}".encode("utf-8")
    digest = hashlib.sha1(seed).hexdigest()
    return f"#{digest[:6]}"


def asset_id_to_label(asset_id: str) -> int:
    num = parse_short_id(asset_id)
    if num is not None:
        return num
    digest = hashlib.sha1(asset_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def label_to_asset_id(label: int) -> str:
    return short_id_from_int(int(label))
