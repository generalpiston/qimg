from __future__ import annotations

import re
import sqlite3

from qimg.util.time import now_iso

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> str:
    return " ".join(t.lower() for t in TOKEN_RE.findall(text))


def path_tokens(path: str) -> tuple[str, str]:
    normalized = path.replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]
    folder = normalized.rsplit("/", 1)[0] if "/" in normalized else ""
    return tokenize(filename), tokenize(folder)


def _context_blob(conn: sqlite3.Connection, asset_id: str) -> str:
    rows = conn.execute(
        """
        SELECT c.text
        FROM contexts c
        JOIN asset_context_effective e ON e.context_id = c.id
        WHERE e.asset_id = ?
        ORDER BY c.virtual_path
        """,
        (asset_id,),
    ).fetchall()
    return " || ".join(str(r["text"]) for r in rows)


def _facts_blob(conn: sqlite3.Connection, asset_id: str) -> str:
    rows = conn.execute(
        """
        SELECT fact_type, key, value_text
        FROM facts
        WHERE asset_id = ? AND fact_type IN ('caption', 'tag', 'object', 'ocr')
        ORDER BY updated_at DESC, id DESC
        """,
        (asset_id,),
    ).fetchall()

    parts: list[str] = []
    for row in rows:
        fact_type = str(row["fact_type"])
        value_text = (row["value_text"] or "").strip()
        key = (row["key"] or "").strip()
        if fact_type in {"tag", "object"}:
            label = value_text or key
            if label:
                parts.append(label)
            continue
        if value_text:
            parts.append(value_text)
    return " || ".join(parts)


def refresh_lexical_row(conn: sqlite3.Connection, asset_id: str) -> None:
    row = conn.execute(
        "SELECT id, rel_path, is_deleted FROM assets WHERE id = ?",
        (asset_id,),
    ).fetchone()

    conn.execute("DELETE FROM lexical_docs WHERE asset_id = ?", (asset_id,))
    conn.execute("DELETE FROM assets_fts WHERE asset_id = ?", (asset_id,))

    if row is None:
        return

    filename_tokens, folder_tokens = path_tokens(str(row["rel_path"]))
    context_text = _context_blob(conn, asset_id)
    facts_text = _facts_blob(conn, asset_id)

    conn.execute(
        """
        INSERT INTO lexical_docs(
          asset_id, filename_tokens, folder_tokens,
          context_text, fact_text, context_only_text, facts_only_text,
          updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            asset_id,
            filename_tokens,
            folder_tokens,
            context_text,
            facts_text,
            context_text.replace("||", " "),
            facts_text.replace("||", " "),
            now_iso(),
        ),
    )

    if int(row["is_deleted"]) == 0:
        conn.execute(
            """
            INSERT INTO assets_fts(asset_id, filename_col, folder_col, context_col, facts_col)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                filename_tokens,
                folder_tokens,
                context_text,
                facts_text,
            ),
        )


def refresh_lexical_for_assets(conn: sqlite3.Connection, asset_ids: set[str]) -> None:
    for asset_id in sorted(asset_ids):
        refresh_lexical_row(conn, asset_id)


def refresh_lexical_for_all(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id FROM assets").fetchall()
    conn.execute("DELETE FROM lexical_docs")
    conn.execute("DELETE FROM assets_fts")
    for row in rows:
        refresh_lexical_row(conn, str(row["id"]))
