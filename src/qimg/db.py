from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Any, Iterator

import numpy as np

from qimg.ids import short_id_from_int, stable_asset_id
from qimg.util.time import now_iso

SETUP_SQL = Path(__file__).resolve().parents[2] / "migrations" / "setup.sql"
SCHEMA_VERSION = 2


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    if not _table_exists(conn, table):
        return False
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(c["name"] == column for c in cols)


def _schema_version(conn: sqlite3.Connection) -> int:
    if not _table_exists(conn, "schema_version"):
        return 0
    row = conn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
    return int(row["version"]) if row else 0


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        """
        INSERT INTO schema_version(id, version, updated_at)
        VALUES(1, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          version=excluded.version,
          updated_at=excluded.updated_at
        """,
        (version, now_iso()),
    )


def _ensure_unique_asset_id(conn: sqlite3.Connection, asset_id: str, collection_id: int, rel_path: str) -> str:
    row = conn.execute("SELECT id, collection_id, rel_path FROM assets WHERE id = ?", (asset_id,)).fetchone()
    if row is None:
        return asset_id
    if int(row["collection_id"]) == int(collection_id) and str(row["rel_path"]) == rel_path:
        return asset_id

    salt = 1
    while True:
        candidate = stable_asset_id(collection_id, rel_path, collision_salt=salt)
        exists = conn.execute("SELECT 1 FROM assets WHERE id = ?", (candidate,)).fetchone()
        if exists is None:
            return candidate
        salt += 1


def _legacy_image_id_to_asset_id(conn: sqlite3.Connection) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if not _table_exists(conn, "images"):
        return mapping

    rows = conn.execute(
        """
        SELECT id, collection_id, rel_path, abs_path, size, mtime, width, height,
               sha256, phash, created_at, updated_at,
               CASE WHEN deleted IS NULL THEN 0 ELSE deleted END AS deleted
        FROM images
        """
    ).fetchall()

    for row in rows:
        old_id = int(row["id"])
        collection_id = int(row["collection_id"])
        rel_path = str(row["rel_path"])

        asset_id = short_id_from_int(old_id)
        asset_id = _ensure_unique_asset_id(conn, asset_id, collection_id, rel_path)

        conn.execute(
            """
            INSERT INTO assets(
              id, collection_id, rel_path, abs_path, media_type, size, mtime,
              width, height, sha256, phash, created_at, updated_at, is_deleted
            ) VALUES (?, ?, ?, ?, 'image', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              collection_id=excluded.collection_id,
              rel_path=excluded.rel_path,
              abs_path=excluded.abs_path,
              size=excluded.size,
              mtime=excluded.mtime,
              width=excluded.width,
              height=excluded.height,
              sha256=excluded.sha256,
              phash=excluded.phash,
              updated_at=excluded.updated_at,
              is_deleted=excluded.is_deleted
            """,
            (
                asset_id,
                collection_id,
                rel_path,
                str(row["abs_path"]),
                int(row["size"]),
                float(row["mtime"]),
                row["width"],
                row["height"],
                row["sha256"],
                row["phash"],
                row["created_at"] or now_iso(),
                row["updated_at"] or now_iso(),
                int(row["deleted"]),
            ),
        )
        mapping[old_id] = asset_id

    return mapping


def _insert_fact(
    conn: sqlite3.Connection,
    asset_id: str,
    fact_type: str,
    key: str | None,
    value_text: str | None,
    value_json: str | None,
    confidence: float | None,
    source: str,
) -> None:
    if value_text is not None and not value_text.strip() and value_json is None:
        return
    ts = now_iso()
    conn.execute(
        """
        INSERT INTO facts(asset_id, fact_type, key, value_text, value_json, confidence, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (asset_id, fact_type, key, value_text, value_json, confidence, source, ts, ts),
    )


def _parse_legacy_tags(tags: str) -> list[str]:
    raw = tags.replace("|", ",").replace(";", ",")
    if "," in raw:
        out = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        out = [x.strip() for x in raw.split() if x.strip()]
    dedup: list[str] = []
    seen: set[str] = set()
    for item in out:
        low = item.lower()
        if low in seen:
            continue
        seen.add(low)
        dedup.append(item)
    return dedup


def _backfill_legacy_facts(conn: sqlite3.Connection, mapping: dict[int, str]) -> None:
    source = "legacy"

    if _table_exists(conn, "images") and _column_exists(conn, "images", "exif_json"):
        rows = conn.execute("SELECT id, exif_json FROM images WHERE exif_json IS NOT NULL AND TRIM(exif_json) != ''").fetchall()
        for row in rows:
            asset_id = mapping.get(int(row["id"]))
            if asset_id is None:
                continue
            exif_json = str(row["exif_json"])
            _insert_fact(conn, asset_id, "exif", "all", None, exif_json, None, source)

    loaded_from_image_text = False
    if _table_exists(conn, "image_text"):
        rows = conn.execute("SELECT image_id, caption, tags, ocr_text FROM image_text").fetchall()
        loaded_from_image_text = bool(rows)
        for row in rows:
            asset_id = mapping.get(int(row["image_id"]))
            if asset_id is None:
                continue

            caption = row["caption"]
            tags = row["tags"]
            ocr_text = row["ocr_text"]
            if caption:
                _insert_fact(conn, asset_id, "caption", None, str(caption), None, None, source)
            if tags:
                for label in _parse_legacy_tags(str(tags)):
                    _insert_fact(conn, asset_id, "tag", label.lower(), label, None, None, source)
            if ocr_text:
                _insert_fact(conn, asset_id, "ocr", None, str(ocr_text), None, None, source)

    if _table_exists(conn, "collection_search_records") and not loaded_from_image_text:
        rows = conn.execute("SELECT image_id, caption, tags, ocr_text FROM collection_search_records").fetchall()
        for row in rows:
            asset_id = mapping.get(int(row["image_id"]))
            if asset_id is None:
                continue
            if row["caption"]:
                _insert_fact(conn, asset_id, "caption", None, str(row["caption"]), None, None, source)
            if row["tags"]:
                for label in _parse_legacy_tags(str(row["tags"])):
                    _insert_fact(conn, asset_id, "tag", label.lower(), label, None, None, source)
            if row["ocr_text"]:
                _insert_fact(conn, asset_id, "ocr", None, str(row["ocr_text"]), None, None, source)


def _backfill_legacy_context_effective(conn: sqlite3.Connection, mapping: dict[int, str]) -> None:
    if not _table_exists(conn, "image_context_map"):
        return
    rows = conn.execute("SELECT image_id, context_id FROM image_context_map").fetchall()
    for row in rows:
        asset_id = mapping.get(int(row["image_id"]))
        if asset_id is None:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO asset_context_effective(asset_id, context_id) VALUES(?, ?)",
            (asset_id, int(row["context_id"])),
        )


def _backfill_legacy_vectors(conn: sqlite3.Connection, mapping: dict[int, str]) -> None:
    if not _table_exists(conn, "image_vectors"):
        return

    has_updated = _column_exists(conn, "image_vectors", "updated_at")
    if has_updated:
        rows = conn.execute("SELECT image_id, model_id, embedding, updated_at FROM image_vectors").fetchall()
    else:
        rows = conn.execute("SELECT image_id, model_id, embedding FROM image_vectors").fetchall()

    for row in rows:
        asset_id = mapping.get(int(row["image_id"]))
        if asset_id is None:
            continue
        updated_at = row["updated_at"] if has_updated else now_iso()
        conn.execute(
            """
            INSERT INTO vectors(asset_id, model_id, embedding, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(asset_id, model_id) DO UPDATE SET
              embedding=excluded.embedding,
              updated_at=excluded.updated_at
            """,
            (asset_id, str(row["model_id"]), row["embedding"], updated_at),
        )


def _backfill_from_legacy_if_needed(conn: sqlite3.Connection) -> None:
    if _schema_version(conn) >= SCHEMA_VERSION:
        return

    if not _table_exists(conn, "images"):
        _set_schema_version(conn, SCHEMA_VERSION)
        return

    # If assets already contain data and no legacy images, this is a fresh DB.
    count_assets = conn.execute("SELECT COUNT(*) AS n FROM assets").fetchone()
    if int(count_assets["n"]) > 0 and not _table_exists(conn, "images"):
        _set_schema_version(conn, SCHEMA_VERSION)
        return

    mapping = _legacy_image_id_to_asset_id(conn)
    if mapping:
        _backfill_legacy_context_effective(conn, mapping)
        _backfill_legacy_facts(conn, mapping)
        _backfill_legacy_vectors(conn, mapping)

    _set_schema_version(conn, SCHEMA_VERSION)


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        sql = SETUP_SQL.read_text()
        with self.connect() as conn:
            conn.executescript(sql)
            _backfill_from_legacy_if_needed(conn)

            # Ensure materialized context/lexical state is up-to-date after migration.
            from qimg.contexts import materialize_context_map
            from qimg.fts import refresh_lexical_for_all

            materialize_context_map(conn)
            refresh_lexical_for_all(conn)


def encode_vector(vec: np.ndarray) -> tuple[bytes, float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr.tobytes(), norm


def decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
