from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qimg.paths import config_root, default_db_path, default_pid_path, default_vector_path


@dataclass(slots=True)
class EmbedConfig:
    model: str = "qimg-encoder-finetuned-clip-vit-large-patch14"
    text_model: str = "qimg-text-encoder-bge-m3"
    device: str = "auto"
    batch_size: int = 16


@dataclass(slots=True)
class SearchConfig:
    rrf_k: int = 60
    query_expansions: int = 2
    top_k_retrieval: int = 100
    rerank_candidates: int = 30
    enable_reranker: bool = False
    w_namefolder: float = 1.0
    w_context: float = 1.2
    w_facts: float = 1.0


@dataclass(slots=True)
class OCRConfig:
    enabled: bool = False


@dataclass(slots=True)
class UIConfig:
    show_logo: bool = True


@dataclass(slots=True)
class AppConfig:
    db_path: Path = field(default_factory=default_db_path)
    vector_dir: Path = field(default_factory=default_vector_path)
    pid_path: Path = field(default_factory=default_pid_path)
    compute_sha256: bool = False
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @property
    def vector_index_path(self) -> Path:
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        return self.vector_dir / "index.bin"

    @property
    def vector_fallback_path(self) -> Path:
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        return self.vector_dir / "vectors.npz"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _to_config(data: dict[str, Any]) -> AppConfig:
    embed = EmbedConfig(**data.get("embed", {}))
    search = SearchConfig(**data.get("search", {}))
    ocr = OCRConfig(**data.get("ocr", {}))
    ui = UIConfig(**data.get("ui", {}))
    return AppConfig(
        db_path=Path(data.get("db_path", str(default_db_path()))).expanduser(),
        vector_dir=Path(data.get("vector_dir", str(default_vector_path()))).expanduser(),
        pid_path=Path(data.get("pid_path", str(default_pid_path()))).expanduser(),
        compute_sha256=bool(data.get("compute_sha256", False)),
        embed=embed,
        search=search,
        ocr=ocr,
        ui=ui,
    )


def default_config_path() -> Path:
    return config_root() / "config.yaml"


def load_config(config_path: Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    path = config_path or default_config_path()
    base: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text())
        if isinstance(loaded, dict):
            base = loaded
    if overrides:
        base = _merge(base, overrides)
    cfg = _to_config(base)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.vector_dir.mkdir(parents=True, exist_ok=True)
    cfg.pid_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg


def write_default_config(path: Path | None = None) -> Path:
    target = path or default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    target.write_text(
        yaml.safe_dump(
            {
                "db_path": str(default_db_path()),
                "vector_dir": str(default_vector_path()),
                "pid_path": str(default_pid_path()),
                "compute_sha256": False,
                "embed": {
                    "model": "qimg-encoder-finetuned-clip-vit-large-patch14",
                    "text_model": "qimg-text-encoder-bge-m3",
                    "device": "auto",
                    "batch_size": 16,
                },
                "search": {
                    "rrf_k": 60,
                    "query_expansions": 2,
                    "top_k_retrieval": 100,
                    "rerank_candidates": 30,
                    "enable_reranker": False,
                    "w_namefolder": 1.0,
                    "w_context": 1.2,
                    "w_facts": 1.0,
                },
                "ocr": {"enabled": False},
                "ui": {"show_logo": True},
            },
            sort_keys=False,
        )
    )
    return target
