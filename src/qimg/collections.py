from __future__ import annotations

from pathlib import Path
import sqlite3

from qimg.models import Collection
from qimg.util.time import now_iso


def add_collection(conn: sqlite3.Connection, name: str, root_path: str, mask: str) -> int:
    root = str(Path(root_path).expanduser().resolve())
    cur = conn.execute(
        "INSERT INTO collections(name, root_path, mask, created_at) VALUES(?, ?, ?, ?)",
        (name, root, mask, now_iso()),
    )
    return int(cur.lastrowid)


def list_collections(conn: sqlite3.Connection) -> list[Collection]:
    rows = conn.execute(
        "SELECT id, name, root_path, mask, created_at FROM collections ORDER BY name"
    ).fetchall()
    return [Collection(**dict(r)) for r in rows]


def get_collection_by_name(conn: sqlite3.Connection, name: str) -> Collection | None:
    row = conn.execute(
        "SELECT id, name, root_path, mask, created_at FROM collections WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        return None
    return Collection(**dict(row))


def rename_collection(conn: sqlite3.Connection, old_name: str, new_name: str) -> None:
    conn.execute("UPDATE collections SET name = ? WHERE name = ?", (new_name, old_name))


def remove_collection(conn: sqlite3.Connection, name: str) -> None:
    conn.execute("DELETE FROM collections WHERE name = ?", (name,))
