from pathlib import Path
import time

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


def _mk_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Content change is enough for incremental update detection via mtime/size.
    payload = f"fake-jpg-{color[0]}-{color[1]}-{color[2]}".encode("utf-8")
    path.write_bytes(payload)


def test_incremental_update_and_embed(tmp_path: Path) -> None:
    root = tmp_path / "imgs"
    img_path = root / "a.jpg"
    _mk_image(img_path, (0, 0, 255))

    svc = QimgService(_cfg(tmp_path))
    svc.collection_add(str(root), name="sample", mask="**/*.{jpg,jpeg,png}")

    stats1 = svc.update()
    assert stats1["added"] == 1
    emb1 = svc.embed()
    assert emb1["embedded"] == 1

    stats2 = svc.update()
    assert stats2["added"] == 0
    assert stats2["changed"] == 0

    time.sleep(1.05)
    _mk_image(img_path, (255, 255, 255))

    stats3 = svc.update()
    assert stats3["changed"] == 1

    emb2 = svc.embed()
    assert emb2["embedded"] >= 1
