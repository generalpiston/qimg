from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from qimg.util.time import now_iso


@dataclass(slots=True)
class ContextEntry:
    id: int
    virtual_path: str
    text: str
    created_at: str


def normalize_virtual_path(path: str) -> str:
    p = path.strip()
    if not p.startswith("qimg://"):
        raise ValueError("virtual path must start with qimg://")
    if p.endswith("/"):
        p = p.rstrip("/")
    return p


def add_context(conn: sqlite3.Connection, virtual_path: str, text: str) -> int:
    vp = normalize_virtual_path(virtual_path)
    cur = conn.execute(
        "INSERT OR REPLACE INTO contexts(virtual_path, text, created_at) VALUES(?, ?, ?)",
        (vp, text.strip(), now_iso()),
    )
    return int(cur.lastrowid)


def list_contexts(conn: sqlite3.Connection, prefix: str | None = None) -> list[ContextEntry]:
    if prefix:
        p = normalize_virtual_path(prefix)
        rows = conn.execute(
            """
            SELECT id, virtual_path, text, created_at
            FROM contexts
            WHERE virtual_path = ? OR virtual_path LIKE ?
            ORDER BY virtual_path
            """,
            (p, f"{p}/%"),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, virtual_path, text, created_at FROM contexts ORDER BY virtual_path"
        ).fetchall()
    return [ContextEntry(**dict(r)) for r in rows]


def remove_context(conn: sqlite3.Connection, virtual_path: str) -> None:
    vp = normalize_virtual_path(virtual_path)
    conn.execute("DELETE FROM contexts WHERE virtual_path = ?", (vp,))


def _asset_virtual_path(collection: str, rel_path: str) -> str:
    if rel_path:
        return f"qimg://{collection}/{rel_path}".rstrip("/")
    return f"qimg://{collection}"


def _matches_context(asset_path: str, context_path: str) -> bool:
    cp = context_path.rstrip("/")
    return asset_path == cp or asset_path.startswith(cp + "/")


def materialize_context_map(conn: sqlite3.Connection, asset_ids: set[str] | None = None) -> None:
    contexts = conn.execute("SELECT id, virtual_path FROM contexts ORDER BY virtual_path").fetchall()

    if asset_ids:
        placeholders = ",".join("?" for _ in asset_ids)
        assets = conn.execute(
            f"""
            SELECT a.id, a.rel_path, c.name AS collection
            FROM assets a
            JOIN collections c ON c.id = a.collection_id
            WHERE a.is_deleted = 0 AND a.id IN ({placeholders})
            """,
            tuple(asset_ids),
        ).fetchall()
        conn.execute(f"DELETE FROM asset_context_effective WHERE asset_id IN ({placeholders})", tuple(asset_ids))
    else:
        assets = conn.execute(
            """
            SELECT a.id, a.rel_path, c.name AS collection
            FROM assets a
            JOIN collections c ON c.id = a.collection_id
            WHERE a.is_deleted = 0
            """
        ).fetchall()
        conn.execute("DELETE FROM asset_context_effective")

    for asset in assets:
        virtual = _asset_virtual_path(str(asset["collection"]), str(asset["rel_path"]))
        for ctx in contexts:
            if _matches_context(virtual, str(ctx["virtual_path"])):
                conn.execute(
                    "INSERT OR IGNORE INTO asset_context_effective(asset_id, context_id) VALUES(?, ?)",
                    (asset["id"], int(ctx["id"])),
                )


def resolve_contexts_for_asset(conn: sqlite3.Connection, asset_id: str) -> list[ContextEntry]:
    rows = conn.execute(
        """
        SELECT c.id, c.virtual_path, c.text, c.created_at
        FROM contexts c
        JOIN asset_context_effective m ON m.context_id = c.id
        WHERE m.asset_id = ?
        ORDER BY c.virtual_path
        """,
        (asset_id,),
    ).fetchall()
    return [ContextEntry(**dict(r)) for r in rows]
