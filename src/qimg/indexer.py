from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

from qimg.contexts import materialize_context_map
from qimg.facts import replace_exif_fact
from qimg.fts import refresh_lexical_for_all
from qimg.ids import stable_asset_id
from qimg.media.image_io import iter_image_files, read_metadata
from qimg.util.time import now_iso


@dataclass(slots=True)
class UpdateStats:
    added: int = 0
    changed: int = 0
    deleted: int = 0
    scanned: int = 0


ASSET_SELECT_SQL = """
SELECT id, rel_path, abs_path, mtime, size, is_deleted
FROM assets
WHERE collection_id = ?
"""


def _new_asset_id(conn: sqlite3.Connection, collection_id: int, rel_path: str) -> str:
    salt = 0
    while True:
        candidate = stable_asset_id(collection_id, rel_path, collision_salt=salt)
        row = conn.execute(
            "SELECT id, collection_id, rel_path FROM assets WHERE id = ?",
            (candidate,),
        ).fetchone()
        if row is None:
            return candidate
        if int(row["collection_id"]) == collection_id and str(row["rel_path"]) == rel_path:
            return candidate
        salt += 1


def _insert_asset(
    conn: sqlite3.Connection,
    collection_id: int,
    rel_path: str,
    abs_path: str,
    mtime: float,
    size: int,
    compute_sha256: bool,
) -> str:
    md = read_metadata(Path(abs_path), compute_hash=compute_sha256)
    now = now_iso()
    asset_id = _new_asset_id(conn, collection_id, rel_path)
    conn.execute(
        """
        INSERT INTO assets(
          id, collection_id, rel_path, abs_path, media_type,
          mtime, size, width, height, sha256, phash,
          created_at, updated_at, is_deleted
        ) VALUES (?, ?, ?, ?, 'image', ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (
            asset_id,
            collection_id,
            rel_path,
            abs_path,
            mtime,
            size,
            md.width,
            md.height,
            md.sha256,
            md.phash,
            now,
            now,
        ),
    )
    replace_exif_fact(conn, asset_id, md.exif_json)
    return asset_id


def _update_asset(
    conn: sqlite3.Connection,
    asset_id: str,
    rel_path: str,
    abs_path: str,
    mtime: float,
    size: int,
    compute_sha256: bool,
) -> None:
    md = read_metadata(Path(abs_path), compute_hash=compute_sha256)
    now = now_iso()
    conn.execute(
        """
        UPDATE assets
        SET rel_path = ?,
            abs_path = ?,
            mtime = ?,
            size = ?,
            width = ?,
            height = ?,
            sha256 = ?,
            phash = ?,
            updated_at = ?,
            is_deleted = 0
        WHERE id = ?
        """,
        (
            rel_path,
            abs_path,
            mtime,
            size,
            md.width,
            md.height,
            md.sha256,
            md.phash,
            now,
            asset_id,
        ),
    )
    replace_exif_fact(conn, asset_id, md.exif_json)


def sync_collections(conn: sqlite3.Connection, compute_sha256: bool = False) -> tuple[UpdateStats, set[str]]:
    stats = UpdateStats()
    changed_ids: set[str] = set()

    collections = conn.execute("SELECT id, name, root_path, mask FROM collections ORDER BY id").fetchall()

    for coll in collections:
        root = Path(str(coll["root_path"]))
        if not root.exists():
            continue

        existing_rows = conn.execute(ASSET_SELECT_SQL, (int(coll["id"]),)).fetchall()
        existing = {str(r["rel_path"]): r for r in existing_rows}
        seen: set[str] = set()

        for fs in iter_image_files(root, str(coll["mask"])):
            stats.scanned += 1
            rel = fs.rel_path
            seen.add(rel)
            row = existing.get(rel)
            if row is None:
                asset_id = _insert_asset(
                    conn,
                    int(coll["id"]),
                    rel,
                    str(fs.abs_path),
                    fs.mtime,
                    fs.size,
                    compute_sha256,
                )
                stats.added += 1
                changed_ids.add(asset_id)
                continue

            if (
                abs(float(row["mtime"]) - float(fs.mtime)) > 1e-6
                or int(row["size"]) != int(fs.size)
                or int(row["is_deleted"]) == 1
            ):
                _update_asset(
                    conn,
                    str(row["id"]),
                    rel,
                    str(fs.abs_path),
                    fs.mtime,
                    fs.size,
                    compute_sha256,
                )
                stats.changed += 1
                changed_ids.add(str(row["id"]))

        for rel, row in existing.items():
            if rel in seen:
                continue
            if int(row["is_deleted"]) == 1:
                continue
            conn.execute(
                "UPDATE assets SET is_deleted = 1, updated_at = ? WHERE id = ?",
                (now_iso(), str(row["id"])),
            )
            stats.deleted += 1
            changed_ids.add(str(row["id"]))

    materialize_context_map(conn)
    refresh_lexical_for_all(conn)
    return stats, changed_ids


def cleanup_orphans(conn: sqlite3.Connection) -> dict[str, int]:
    out: dict[str, int] = {}

    deleted_assets = conn.execute("DELETE FROM assets WHERE is_deleted = 1").rowcount
    out["deleted_assets"] = int(deleted_assets)

    orphan_context_effective = conn.execute(
        "DELETE FROM asset_context_effective WHERE asset_id NOT IN (SELECT id FROM assets) OR context_id NOT IN (SELECT id FROM contexts)"
    ).rowcount
    out["orphan_context_links"] = int(orphan_context_effective)

    orphan_facts = conn.execute("DELETE FROM facts WHERE asset_id NOT IN (SELECT id FROM assets)").rowcount
    out["orphan_facts"] = int(orphan_facts)

    orphan_lexical = conn.execute("DELETE FROM lexical_docs WHERE asset_id NOT IN (SELECT id FROM assets)").rowcount
    out["orphan_lexical_docs"] = int(orphan_lexical)

    orphan_fts = conn.execute("DELETE FROM assets_fts WHERE asset_id NOT IN (SELECT id FROM assets)").rowcount
    out["orphan_fts_rows"] = int(orphan_fts)

    orphan_vectors = conn.execute("DELETE FROM vectors WHERE asset_id NOT IN (SELECT id FROM assets)").rowcount
    out["orphan_vectors"] = int(orphan_vectors)

    refresh_lexical_for_all(conn)
    return out
