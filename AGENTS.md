# AGENTS.md

## Purpose

Guidance for code agents working in this repository. Follow existing patterns over personal preference.

## Stack

- Python 3.13
- Package layout under `src/qimg`
- CLI built with `typer`
- Tests with `pytest`
- Optional ML/runtime deps behind extras

## Style Conventions

- Use `from __future__ import annotations` in Python modules.
- Prefer explicit typing throughout (`dict[str, Any]`, `list[str]`, unions with `|`).
- Use `@dataclass(slots=True)` for lightweight data containers; use `frozen=True` when immutable.
- Keep functions focused and deterministic where possible.
- Prefer small helper functions for repeated logic.
- Use `Path` from `pathlib` for filesystem paths.
- Use 4-space indentation and keep formatting close to current files.
- Keep comments sparse and only for non-obvious behavior.

## Error Handling

- Raise explicit exceptions with actionable messages.
- Fail early on missing config/model/data requirements.
- For optional dependencies, keep graceful fallbacks where already used.

## CLI and Output

- Route user actions through `qimg.service` instead of duplicating logic in CLI handlers.
- Keep CLI output consistent with existing modes:
  - human-readable rich output
  - `--json` for structured output
  - `--files` where applicable
- Preserve existing command names/options and backward-compatible behavior.

## Models and MCP

- Runtime model management belongs in `src/qimg/model_manager.py`.
- Model artifacts belong under `models/`; do not add new executable scripts there.
- MCP protocol/tool names live in `src/qimg/mcp/protocol.py`; keep changes backward compatible.

## Testing Expectations

- Add/update tests for behavior changes under `tests/`.
- Follow current test style: small fixtures/helpers, direct assertions, minimal mocking.
- Prefer deterministic tests that avoid network access.
- Run:
  - `uv run pytest -q`
  - or targeted tests for touched areas.

## Dependency and License Hygiene

- Keep heavy dependencies optional via extras when possible.
- Do not introduce GPL/AGPL-licensed direct dependencies without explicit approval.
- Keep metadata aligned (`pyproject.toml`, `LICENSE`, docs).

## Change Discipline

- Make minimal, scoped edits.
- Do not rename public APIs/commands without clear migration handling.
- Update docs when user-facing behavior changes.
