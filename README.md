# qimg

`qimg` is a local-first image retrieval CLI and MCP server with strict separation between:

1. Contexts (human-authored priors)
2. Facts (machine-extracted attributes)
3. Embeddings (dense vectors)
4. Core metadata (file/path/mtime/size/dimensions/hashes)

This separation keeps retrieval interpretable and lets you tune ranking behavior channel-by-channel.

## Requirements

- Python `3.13`
- `uv`
- macOS/Linux (Windows untested)

## Quickstart

```bash
cd qimg
uv python install 3.13
uv sync --python 3.13 --extra test
```

Optional extras:

```bash
uv sync --python 3.13 --extra test --extra hnsw
uv sync --python 3.13 --extra test --extra encoder
uv sync --python 3.13 --extra test --extra ocr
```

Note: semantic retrieval uses vendored encoder models under `models/`.  
Fallback embedder is disabled by default for quality. You can only force it with `QIMG_ALLOW_FALLBACK=1`.
If a required built-in model is missing, qimg now syncs it on-demand at first use.

Download vendored runtime models:

```bash
uv run --extra encoder qimg models download
# or only one channel:
uv run --extra encoder qimg models download --text
# offline from local HF cache:
uv run --extra encoder qimg models download --from-cache
```

Initialize default config:

```bash
uv run qimg init-config
# optional: initialize config and immediately download models:
uv run --extra encoder qimg init-config --download-models
```

## Concepts

### Contexts (human)

- Added by users, attached to virtual paths: `qimg://<collection>/<prefix>`
- Inherited down the path tree
- Stable semantic priors

```bash
uv run qimg context add qimg://photos "Family photos"
uv run qimg context add qimg://photos/2024 "Japan trip"
uv run qimg context list
```

### Facts (machine)

- Mutable/recomputable with provenance (`source`)
- Includes: `caption`, `tag`, `object`, `ocr`, `exif`, `derived`
- Never stored as contexts
- `object` facts come from a vendored detector model (DETR), not path heuristics

```bash
uv run qimg facts extract --caption --tags --objects --ocr
uv run qimg facts ls '#00000a'
uv run qimg facts rm --source heuristic_labels@1.0
```

For model-based `--objects` extraction, run with encoder deps installed (`uv run --extra encoder ...`) and vendor the detector model under `models/`.
You can vendor all runtime models with `uv run --extra encoder qimg models download`.

## Basic Flow

```bash
uv run qimg collection add ./sample_images --name sample --mask "**/*.{jpg,jpeg,png,webp,heic}"
uv run qimg context add qimg://sample "Sample images"
uv run qimg update
uv run qimg facts extract --caption --tags
uv run qimg embed
uv run qimg search "kitchen" -n 5
uv run qimg query "minimalist white kitchen" -n 5 --debug
```

## Commands

### Collections

```bash
uv run qimg collection list
uv run qimg collection rename sample photos
uv run qimg collection remove photos
```

### Models

```bash
uv run --extra encoder qimg models download
uv run --extra encoder qimg models download --image
uv run --extra encoder qimg models download --from-cache
```

Model syncing is also automatic when a command needs a built-in model:
- image/text encoders sync on first `embed` / `vsearch` / `query`
- object detector syncs on first `facts extract --objects`

### Indexing

```bash
uv run qimg update
```

`update` syncs core `assets` metadata and deterministic EXIF facts only. It does not generate captions/tags/OCR.

### Facts

```bash
uv run qimg facts extract --caption --tags --objects --ocr
uv run qimg facts ls qimg://photos/2024/trip.jpg
uv run qimg facts rm --source legacy
```

If no extract flags are provided (`qimg facts extract`), qimg runs all extraction channels by default.

### Embeddings

```bash
uv run qimg embed
uv run qimg embed --images
uv run qimg embed --ocr-text
```

`embed` defaults to embedding both image content and OCR text (when OCR facts exist).
Use flags to limit scope to only one channel.
Image model and text model are independently configurable in `config.yaml`:

```yaml
embed:
  model: qimg-encoder-finetuned-clip-vit-large-patch14
  text_model: qimg-text-encoder-bge-m3
ui:
  show_logo: true
```

Set `ui.show_logo: false` to disable the startup logo.

### Retrieval

- `qimg search "..."`: lexical only (contexts/facts/name-folder split)
- `qimg vsearch "..."`: vector only
- `qimg query "..."`: hybrid (query expansion + lexical + vector + RRF + optional rerank)

Defaults:
- `vsearch` and `query` apply `--min-score 0.75` unless you override with `--min-score`.

Shared filters:

- `-n/--num`
- `-c/--collection`
- `--path-prefix`
- `--after YYYY-MM-DD`
- `--before YYYY-MM-DD`
- `--min-width`
- `--min-height`
- `--all`
- `--min-score`

Output modes:

- default: path/id/score with context, caption, key EXIF
- `--files`: TSV lines
- `--json`: structured agent-first output with separated `contexts` and `facts`
- `--debug` (search/query/vsearch/get/multi-get/similar): score channel breakdown

## Ranking Semantics

### Lexical search

`search` computes split lexical signals:

- `score_namefolder`
- `score_context`
- `score_facts`

Combined score:

```text
total_lex_score = w_namefolder*score_namefolder + w_context*score_context + w_facts*score_facts
```

Default weights:

- `w_namefolder = 1.0`
- `w_context = 1.2`
- `w_facts = 1.0`

### Hybrid query

`query` uses deterministic expansion:

- original query (`x2` weight)
- expansion #1 (object/detail oriented)
- expansion #2 (style/scene oriented)

For each expanded query:

- lexical retrieval (with context/fact channel bias)
- vector retrieval

Fusion:

- Reciprocal Rank Fusion (`k=60`) with top-rank bonuses
- Optional rerank over contexts + facts + metadata

## JSON Output Shape

Each result includes:

- `contexts`: `[ { virtual_path, text } ]`
- `facts`: `{ caption, tags, objects, ocr, exif, derived }`
- `metadata`: `{ collection, rel_path, abs_path, width, height, size, mtime, date, camera }`
- `scores`: `{ overall, rrf, lexical_total, lexical_namefolder, lexical_context, lexical_facts, lexical_ocr, vector, vector_image, vector_ocr, rerank, contributions }`

## MCP

### stdio

```bash
uv run qimg mcp
```

Example request:

```json
{"tool":"qimg_deep_search","args":{"query":"minimalist white kitchen","num":5}}
```

### HTTP

```bash
uv run qimg mcp --http --host 127.0.0.1 --port 8181
curl http://127.0.0.1:8181/health
curl -X POST http://127.0.0.1:8181/mcp -H 'content-type: application/json' -d '{"tool":"qimg_status"}'
```

### Daemon

```bash
uv run qimg mcp --http --daemon
uv run qimg mcp stop
```

### Client Integration Docs

- Claude Code: `integrations/claude-code/README.md`
- Codex: `integrations/codex/README.md`

Additional MCP tools:

- `qimg_facts_extract`
- `qimg_facts_get`

## Storage Model

Schema lives in `migrations/setup.sql`.

Core tables:

- `collections`
- `assets`
- `contexts`
- `asset_context_effective`
- `facts`
- `lexical_docs`
- `assets_fts`
- `vectors`
- `llm_cache`

## Migration

At startup, `qimg` initializes the new schema and auto-migrates legacy databases (`images`, `image_text`, `image_context_map`, `image_vectors`) into:

- `assets`
- `facts` (`source=legacy`)
- `asset_context_effective`
- `vectors`

Then it rebuilds materialized lexical/context state.

## Testing

```bash
uv run pytest
```

Includes tests for:

- strict context inheritance
- facts provenance/source removal
- split FTS scoring
- JSON query output schema
- legacy migration backfill
