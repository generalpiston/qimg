from pathlib import Path

from qimg.config import AppConfig, EmbedConfig, OCRConfig, SearchConfig
from qimg.service import QimgService, SearchFilters
from qimg.util.time import now_iso


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        db_path=tmp_path / "qimg.sqlite3",
        vector_dir=tmp_path / "vectors",
        pid_path=tmp_path / "mcp.pid",
        embed=EmbedConfig(model="fallback-clip", device="cpu"),
        search=SearchConfig(enable_reranker=False),
        ocr=OCRConfig(enabled=False),
    )


def _mk_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-jpg-bytes-ocr-vector")


def test_vsearch_includes_ocr_vectors(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    img = root / "scan.jpg"
    _mk_image(img)

    svc = QimgService(_cfg(tmp_path))
    svc.collection_add(str(root), name="docs", mask="**/*.{jpg,jpeg,png}")
    svc.update()
    item = svc.get(str(img))
    assert item is not None
    asset_id = item["id"]

    with svc.db.connect() as conn:
        conn.execute(
            """
            INSERT INTO facts(asset_id, fact_type, key, value_text, value_json, confidence, source, created_at, updated_at)
            VALUES (?, 'ocr', NULL, ?, NULL, NULL, 'test_ocr@1.0', ?, ?)
            """,
            (asset_id, "invoice total due payment 4289", now_iso(), now_iso()),
        )

    # Only OCR text embeddings are needed for this retrieval path.
    emb = svc.embed(include_images=False, include_ocr_text=True)
    assert emb["embedded_ocr_text"] >= 1

    rows = svc.vsearch("invoice payment due", SearchFilters(num=5))
    assert rows
    top = rows[0]
    assert top["id"] == asset_id
    assert top["scores"]["vector_ocr"] is not None
    assert top["scores"]["lexical_ocr"] is not None
    assert float(top["scores"]["lexical_ocr"]) > 0.0

    # Hybrid query should preserve high confidence for a dominant OCR/vector match.
    qrows = svc.query("invoice payment due", SearchFilters(num=5))
    assert qrows
    qtop = qrows[0]
    assert qtop["id"] == asset_id
    assert qtop["scores"]["vector_ocr"] is not None
    assert float(qtop["scores"]["overall"]) >= 0.8
