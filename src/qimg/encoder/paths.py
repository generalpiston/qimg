from __future__ import annotations

import os
from pathlib import Path


def models_root() -> Path:
    env = os.environ.get("QIMG_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return (parent / "models").resolve()
    return (Path.cwd() / "models").resolve()


def model_dir(model_name: str) -> Path:
    return models_root() / model_name

