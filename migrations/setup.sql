-- qimg schema bootstrap (normalized architecture).

CREATE TABLE IF NOT EXISTS collections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  root_path TEXT NOT NULL,
  mask TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
  id TEXT PRIMARY KEY,
  collection_id INTEGER NOT NULL,
  rel_path TEXT NOT NULL,
  abs_path TEXT NOT NULL UNIQUE,
  media_type TEXT NOT NULL DEFAULT 'image',
  size INTEGER NOT NULL,
  mtime REAL NOT NULL,
  width INTEGER,
  height INTEGER,
  sha256 TEXT,
  phash TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  is_deleted INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY(collection_id) REFERENCES collections(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS contexts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  virtual_path TEXT NOT NULL UNIQUE,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS asset_context_effective (
  asset_id TEXT NOT NULL,
  context_id INTEGER NOT NULL,
  PRIMARY KEY(asset_id, context_id),
  FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE,
  FOREIGN KEY(context_id) REFERENCES contexts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  asset_id TEXT NOT NULL,
  fact_type TEXT NOT NULL CHECK (fact_type IN ('caption', 'tag', 'object', 'ocr', 'exif', 'derived')),
  key TEXT,
  value_text TEXT,
  value_json TEXT,
  confidence REAL,
  source TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS lexical_docs (
  asset_id TEXT PRIMARY KEY,
  filename_tokens TEXT NOT NULL DEFAULT '',
  folder_tokens TEXT NOT NULL DEFAULT '',
  context_text TEXT NOT NULL DEFAULT '',
  fact_text TEXT NOT NULL DEFAULT '',
  context_only_text TEXT NOT NULL DEFAULT '',
  facts_only_text TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL,
  FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS assets_fts USING fts5(
  asset_id UNINDEXED,
  filename_col,
  folder_col,
  context_col,
  facts_col
);

CREATE TABLE IF NOT EXISTS vectors (
  asset_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  embedding BLOB NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(asset_id, model_id),
  FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS llm_cache (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_version (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  version INTEGER NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_assets_collection_rel ON assets(collection_id, rel_path);
CREATE INDEX IF NOT EXISTS idx_assets_deleted ON assets(is_deleted);
CREATE INDEX IF NOT EXISTS idx_contexts_virtual_path ON contexts(virtual_path);
CREATE INDEX IF NOT EXISTS idx_asset_context_effective_asset ON asset_context_effective(asset_id);
CREATE INDEX IF NOT EXISTS idx_facts_asset_type ON facts(asset_id, fact_type);
CREATE INDEX IF NOT EXISTS idx_facts_source ON facts(source);
CREATE INDEX IF NOT EXISTS idx_vectors_model ON vectors(model_id);
