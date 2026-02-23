from pathlib import Path

from qimg.config import AppConfig, EmbedConfig, OCRConfig, SearchConfig
from qimg.service import QimgService


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
    path.write_bytes(b"fake-jpg-bytes-context")


def test_context_inheritance_strict(tmp_path: Path) -> None:
    root = tmp_path / "photos"
    _mk_image(root / "2024" / "trip.jpg")

    svc = QimgService(_cfg(tmp_path))
    svc.collection_add(str(root), name="photos", mask="**/*.{jpg,jpeg,png}")
    svc.context_add("qimg://photos", "Family photos")
    svc.context_add("qimg://photos/2024", "Japan trip")
    svc.update()
    svc.facts_extract(extract_caption=True, extract_tags=True, extract_objects=True, extract_ocr=False)

    item = svc.get("qimg://photos/2024/trip.jpg")
    assert item is not None

    contexts = item["contexts"]
    assert [c["text"] for c in contexts] == ["Family photos", "Japan trip"]

    facts = item["facts"]
    tag_labels = [t["label"] for t in facts["tags"]]
    assert "Family photos" not in tag_labels
    assert "Japan trip" not in tag_labels
