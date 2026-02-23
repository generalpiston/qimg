from __future__ import annotations

import sys


def _has_mps(torch_module: object) -> bool:
    backends = getattr(torch_module, "backends", None)
    if backends is None:
        return False
    mps = getattr(backends, "mps", None)
    if mps is None:
        return False
    is_available = getattr(mps, "is_available", None)
    if not callable(is_available):
        return False
    return bool(is_available())


def resolve_device(requested: str | None = None) -> str:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for vendored local image encoder") from exc

    req = (requested or "auto").strip().lower()
    if req not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {requested}")

    if req == "cpu":
        return "cpu"
    if req == "cuda":
        if not bool(torch.cuda.is_available()):
            raise RuntimeError("cuda requested but not available")
        return "cuda"
    if req == "mps":
        if not _has_mps(torch):
            raise RuntimeError("mps requested but not available")
        return "mps"

    if sys.platform == "darwin" and _has_mps(torch):
        return "mps"
    if sys.platform.startswith("linux") and bool(torch.cuda.is_available()):
        return "cuda"
    if bool(torch.cuda.is_available()):
        return "cuda"
    return "cpu"

