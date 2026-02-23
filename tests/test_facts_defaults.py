from pathlib import Path

from qimg.config import AppConfig, EmbedConfig, OCRConfig, SearchConfig
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
    path.write_bytes(b"fake-jpg-bytes-default-facts")


def test_facts_extract_defaults_to_all_channels(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "imgs"
    img = root / "receipt-2026.jpg"
    _mk_image(img)

    class _StubDetector:
        source = "object_detector:test@1.0"

        def detect(self, path: Path):
            return [type("Obj", (), {"label": "paper", "confidence": 0.92, "bbox_xyxy": (1.0, 1.0, 9.0, 9.0)})()]

    monkeypatch.setattr("qimg.facts._get_object_detector", lambda: _StubDetector())
    monkeypatch.setattr("qimg.facts.ocr_runtime_available", lambda: True)
    monkeypatch.setattr("qimg.facts._extract_ocr_text", lambda p: "invoice total 42")

    svc = QimgService(_cfg(tmp_path))
    svc.collection_add(str(root), name="sample", mask="**/*.{jpg,jpeg,png}")
    svc.update()

    # No flags selected -> should run all channels.
    extracted = svc.facts_extract(
        extract_caption=False,
        extract_tags=False,
        extract_objects=False,
        extract_ocr=False,
    )

    assert extracted["captions"] > 0
    assert extracted["tags"] > 0
    assert extracted["objects"] > 0
    assert extracted["ocr"] > 0

