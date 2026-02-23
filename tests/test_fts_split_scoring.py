from pathlib import Path

from qimg.config import AppConfig, EmbedConfig, OCRConfig, SearchConfig
from qimg.fts import refresh_lexical_for_all
from qimg.service import QimgService, SearchFilters
from qimg.util.time import now_iso


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        db_path=tmp_path / "qimg.sqlite3",
        vector_dir=tmp_path / "vectors",
        pid_path=tmp_path / "mcp.pid",
        embed=EmbedConfig(model="fallback-clip", device="cpu"),
        search=SearchConfig(w_namefolder=1.0, w_context=1.2, w_facts=1.0),
        ocr=OCRConfig(enabled=False),
    )


def _mk_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"bytes-{path.name}".encode("utf-8"))


def test_fts_split_scoring(tmp_path: Path) -> None:
    root = tmp_path / "imgs"
    _mk_image(root / "ctx" / "alpha.jpg")
    _mk_image(root / "facts" / "beta.jpg")

    svc = QimgService(_cfg(tmp_path))
    svc.collection_add(str(root), name="sample", mask="**/*.{jpg,jpeg,png}")
    svc.context_add("qimg://sample/ctx", "galaxy")
    svc.update()

    facts_item = svc.get("qimg://sample/facts/beta.jpg")
    assert facts_item is not None
    with svc.db.connect() as conn:
        conn.execute(
            """
            INSERT INTO facts(asset_id, fact_type, key, value_text, value_json, confidence, source, created_at, updated_at)
            VALUES (?, 'tag', 'galaxy', 'galaxy', NULL, 0.9, 'test@1.0', ?, ?)
            """,
            (facts_item["id"], now_iso(), now_iso()),
        )
        refresh_lexical_for_all(conn)

    rows = svc.search("galaxy", SearchFilters(num=10))

    by_rel = {r["rel_path"]: r for r in rows}
    assert "ctx/alpha.jpg" in by_rel
    assert "facts/beta.jpg" in by_rel

    context_row = by_rel["ctx/alpha.jpg"]
    facts_row = by_rel["facts/beta.jpg"]

    assert context_row["scores"]["lexical_context"] > 0.0
    assert facts_row["scores"]["lexical_facts"] > 0.0
    assert context_row["scores"]["overall"] >= facts_row["scores"]["overall"]
