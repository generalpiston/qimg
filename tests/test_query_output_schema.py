from pathlib import Path

from qimg.config import AppConfig, EmbedConfig, OCRConfig, SearchConfig
from qimg.service import QimgService, SearchFilters


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
    path.write_bytes(b"fake-jpg-bytes-query")


def test_query_output_schema(tmp_path: Path) -> None:
    root = tmp_path / "photos"
    _mk_image(root / "sunset-beach.jpg")

    svc = QimgService(_cfg(tmp_path))
    svc.collection_add(str(root), name="photos", mask="**/*.{jpg,jpeg,png}")
    svc.context_add("qimg://photos", "vacation album")
    svc.update()
    svc.facts_extract(extract_caption=True, extract_tags=True, extract_objects=True, extract_ocr=False)

    rows = svc.query("sunset", SearchFilters(num=3))
    assert rows

    row = rows[0]
    assert {"contexts", "facts", "metadata", "scores"}.issubset(row.keys())

    assert isinstance(row["contexts"], list)
    if row["contexts"]:
        assert {"virtual_path", "text"}.issubset(row["contexts"][0].keys())

    assert {"caption", "tags", "objects", "ocr", "exif", "derived"}.issubset(row["facts"].keys())
    assert {"rrf", "lexical_total", "lexical_context", "lexical_facts", "vector", "rerank"}.issubset(
        row["scores"].keys()
    )
