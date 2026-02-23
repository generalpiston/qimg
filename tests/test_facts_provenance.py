from pathlib import Path

from qimg.config import AppConfig, EmbedConfig, OCRConfig, SearchConfig
from qimg.facts import LABEL_SOURCE
from qimg.service import QimgService


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        db_path=tmp_path / "qimg.sqlite3",
        vector_dir=tmp_path / "vectors",
        pid_path=tmp_path / "mcp.pid",
        embed=EmbedConfig(model="fallback-clip", device="cpu"),
        search=SearchConfig(),
        ocr=OCRConfig(enabled=False),
    )


def _mk_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-jpg-bytes-facts")


def test_facts_provenance(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "imgs"
    img = root / "office-desk.jpg"
    _mk_image(img)

    class _StubDetector:
        source = "object_detector:test@1.0"

        def detect(self, path: Path):  # pragma: no cover - exercised via facts extraction
            return [
                type("Obj", (), {"label": "desk", "confidence": 0.93, "bbox_xyxy": (1.0, 2.0, 3.0, 4.0)})(),
                type("Obj", (), {"label": "laptop", "confidence": 0.82, "bbox_xyxy": (5.0, 6.0, 7.0, 8.0)})(),
            ]

    monkeypatch.setattr("qimg.facts._get_object_detector", lambda: _StubDetector())

    svc = QimgService(_cfg(tmp_path))
    svc.collection_add(str(root), name="sample", mask="**/*.{jpg,jpeg,png}")
    svc.update()

    extracted = svc.facts_extract(
        extract_caption=False,
        extract_tags=True,
        extract_objects=True,
        extract_ocr=False,
    )
    assert extracted["tags"] > 0
    assert extracted["objects"] > 0

    item = svc.get(str(img))
    assert item is not None
    asset_id = item["id"]

    facts = svc.facts_ls(asset_id)
    tag_rows = [f for f in facts if f["fact_type"] == "tag"]
    object_rows = [f for f in facts if f["fact_type"] == "object"]
    assert tag_rows and object_rows
    assert all(f["source"] == LABEL_SOURCE for f in tag_rows)
    assert all(f["confidence"] is not None for f in tag_rows)
    assert all(f["source"] == "object_detector:test@1.0" for f in object_rows)
    assert all(f["confidence"] is not None for f in object_rows)

    rm_stats = svc.facts_rm(source=LABEL_SOURCE)
    assert rm_stats["removed"] >= len(tag_rows)

    facts_after = svc.facts_ls(asset_id)
    assert all(f["source"] != LABEL_SOURCE for f in facts_after)
    assert any(f["source"] == "object_detector:test@1.0" for f in facts_after)
