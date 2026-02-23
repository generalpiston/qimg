from pathlib import Path

from qimg.db import Database


REQUIRED_TABLES = {
    "collections",
    "assets",
    "contexts",
    "asset_context_effective",
    "facts",
    "lexical_docs",
    "assets_fts",
    "vectors",
    "llm_cache",
    "schema_version",
}


def test_schema_tables_exist(tmp_path: Path) -> None:
    db = Database(tmp_path / "qimg.sqlite3")
    db.initialize()
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        ).fetchall()
    names = {r["name"] for r in rows}
    assert REQUIRED_TABLES.issubset(names)
