from pathlib import Path
import sqlite3

import numpy as np

from qimg.db import Database


LEGACY_SQL = """
CREATE TABLE collections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  root_path TEXT NOT NULL,
  mask TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  collection_id INTEGER NOT NULL,
  rel_path TEXT NOT NULL,
  abs_path TEXT NOT NULL UNIQUE,
  mtime REAL NOT NULL,
  size INTEGER NOT NULL,
  width INTEGER,
  height INTEGER,
  exif_json TEXT,
  sha256 TEXT,
  phash TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  deleted INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE contexts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  virtual_path TEXT NOT NULL UNIQUE,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE image_context_map (
  image_id INTEGER NOT NULL,
  context_id INTEGER NOT NULL,
  PRIMARY KEY(image_id, context_id)
);

CREATE TABLE image_text (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  image_id INTEGER NOT NULL UNIQUE,
  filename_tokens TEXT,
  folder_tokens TEXT,
  caption TEXT,
  tags TEXT,
  ocr_text TEXT
);

CREATE TABLE image_vectors (
  image_id INTEGER NOT NULL PRIMARY KEY,
  model_id TEXT NOT NULL,
  embedding BLOB NOT NULL,
  norm REAL NOT NULL,
  updated_at TEXT NOT NULL
);
"""


def test_migration_backfill(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript(LEGACY_SQL)

    conn.execute(
        "INSERT INTO collections(id, name, root_path, mask, created_at) VALUES(1, 'photos', '/tmp/photos', '**/*.jpg', '2024-01-01T00:00:00Z')"
    )
    conn.execute(
        """
        INSERT INTO images(
          id, collection_id, rel_path, abs_path, mtime, size, width, height,
          exif_json, sha256, phash, created_at, updated_at, deleted
        ) VALUES (
          10, 1, 'a.jpg', '/tmp/photos/a.jpg', 1700000000.0, 123,
          800, 600,
          '{"datetime":"2024:01:02 03:04:05","camera_model":"Canon R6"}',
          'abc', 'ffff', '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', 0
        )
        """
    )
    conn.execute(
        "INSERT INTO contexts(id, virtual_path, text, created_at) VALUES(5, 'qimg://photos', 'Family photos', '2024-01-01T00:00:00Z')"
    )
    conn.execute("INSERT INTO image_context_map(image_id, context_id) VALUES(10, 5)")
    conn.execute(
        "INSERT INTO image_text(image_id, filename_tokens, folder_tokens, caption, tags, ocr_text) VALUES(10, 'a jpg', '', 'legacy caption', 'cat,dog', 'hello world')"
    )

    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    conn.execute(
        "INSERT INTO image_vectors(image_id, model_id, embedding, norm, updated_at) VALUES(10, 'legacy-model', ?, 1.0, '2024-01-01T00:00:00Z')",
        (vec.tobytes(),),
    )
    conn.commit()
    conn.close()

    db = Database(db_path)
    db.initialize()

    with db.connect() as conn2:
        asset = conn2.execute("SELECT id, rel_path FROM assets WHERE rel_path = 'a.jpg'").fetchone()
        assert asset is not None
        assert asset["id"] == "#00000a"

        facts = conn2.execute(
            "SELECT fact_type, source FROM facts WHERE asset_id = ? ORDER BY fact_type",
            (asset["id"],),
        ).fetchall()
        fact_types = [r["fact_type"] for r in facts]
        assert {"caption", "tag", "ocr", "exif"}.issubset(set(fact_types))
        assert all(r["source"] == "legacy" for r in facts)

        context_link = conn2.execute(
            "SELECT 1 FROM asset_context_effective WHERE asset_id = ? AND context_id = 5",
            (asset["id"],),
        ).fetchone()
        assert context_link is not None

        vector = conn2.execute(
            "SELECT model_id FROM vectors WHERE asset_id = ?",
            (asset["id"],),
        ).fetchone()
        assert vector is not None
        assert vector["model_id"] == "legacy-model"
