from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None  # type: ignore[assignment]
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer  # type: ignore

from qimg.encoder.device import resolve_device
from qimg.model_manager import get_model_manager


@contextmanager
def _silence_fds():
    out_fd: int | None = None
    err_fd: int | None = None
    devnull_fd: int | None = None
    try:
        out_fd = os.dup(1)
        err_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        if out_fd is not None:
            os.dup2(out_fd, 1)
            os.close(out_fd)
        if err_fd is not None:
            os.dup2(err_fd, 2)
            os.close(err_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)


def _model_load_quiet_ctx():
    if os.environ.get("QIMG_VERBOSE_MODEL_LOAD", "").strip() == "1":
        return nullcontext()
    sink = io.StringIO()
    stack = ExitStack()
    stack.enter_context(_silence_fds())
    stack.enter_context(redirect_stdout(sink))
    stack.enter_context(redirect_stderr(sink))
    return stack


class LocalImageEncoder:
    """Local Hugging Face image-text encoder loaded from vendored models/<model>/."""

    def __init__(
        self,
        model: str = "qimg-encoder-finetuned-clip-vit-large-patch14",
        dim: int | None = None,
        device: str = "auto",
    ):
        self.model = (model or "qimg-encoder-finetuned-clip-vit-large-patch14").strip()
        self.device = resolve_device(device)
        self._model_manager = get_model_manager()
        self._model_manager.sync_required([self.model])
        self.model_path = self._model_manager.model_dir(self.model)
        self._model, self._processor, self._source_dir = self._load_model()
        self._model.eval()
        emb_dim = int(getattr(self._model.config, "projection_dim", 0) or 0)
        if emb_dim <= 0:
            raise ValueError("projection_dim missing from model config")
        self.dim = emb_dim
        if dim is not None and int(dim) != self.dim:
            raise ValueError(f"encoder dim mismatch: requested {dim}, model provides {self.dim}")

    def _load_model(self) -> tuple[Any, Any, Path]:
        manifest_path = self.model_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"model manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())

        finetuned = self.model_path / str(manifest.get("finetuned_subdir", "finetuned"))
        base = self.model_path / str(manifest.get("base_subdir", "base"))
        source_dir = finetuned if self._is_checkpoint_dir(finetuned) else base
        if not source_dir.exists():
            raise FileNotFoundError(
                f"no local checkpoint found under {finetuned} or {base}; "
                "run `qimg models download` to vendor required model checkpoints"
            )

        with _model_load_quiet_ctx():
            model = AutoModel.from_pretrained(str(source_dir)).to(self.device)
            processor = AutoProcessor.from_pretrained(str(source_dir))
        return model, processor, source_dir

    @staticmethod
    def _is_checkpoint_dir(path: Path) -> bool:
        return (path / "config.json").exists()

    def encode(self, path: Path) -> np.ndarray:
        if Image is None:
            raise RuntimeError("Pillow is required for image embedding")
        with Image.open(path) as img:
            image = img.convert("RGB")
        inputs = self._processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            if hasattr(self._model, "get_image_features"):
                out = self._model.get_image_features(**inputs)
            else:
                out = self._model(**inputs)
            emb = self._extract_embedding(out, kind="image")
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return emb[0].detach().to("cpu").numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            if hasattr(self._model, "get_text_features"):
                out = self._model.get_text_features(**inputs)
            else:
                out = self._model(**inputs)
            emb = self._extract_embedding(out, kind="text")
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return emb[0].detach().to("cpu").numpy().astype(np.float32)

    @staticmethod
    def _extract_embedding(out: Any, kind: str) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out
        if kind == "image":
            if hasattr(out, "image_embeds") and getattr(out, "image_embeds") is not None:
                return getattr(out, "image_embeds")
        if kind == "text":
            if hasattr(out, "text_embeds") and getattr(out, "text_embeds") is not None:
                return getattr(out, "text_embeds")
        if hasattr(out, "pooler_output") and getattr(out, "pooler_output") is not None:
            return getattr(out, "pooler_output")
        if hasattr(out, "last_hidden_state") and getattr(out, "last_hidden_state") is not None:
            h = getattr(out, "last_hidden_state")
            if h.ndim == 3:
                return h.mean(dim=1)
            return h
        raise TypeError(f"unsupported model output type for {kind} embedding: {type(out)!r}")


class LocalTextEncoder:
    """Local text embedding encoder loaded from vendored models/<model>/."""

    def __init__(
        self,
        model: str = "qimg-text-encoder-bge-m3",
        dim: int | None = None,
        device: str = "auto",
    ):
        self.model = (model or "qimg-text-encoder-bge-m3").strip()
        self.device = resolve_device(device)
        self._model_manager = get_model_manager()
        self._model_manager.sync_required([self.model])
        self.model_path = self._model_manager.model_dir(self.model)
        self._model, self._tokenizer, self._source_dir = self._load_model()
        self._model.eval()
        self.max_length = self._resolve_max_length()

        emb_dim = int(getattr(self._model.config, "hidden_size", 0) or getattr(self._model.config, "projection_dim", 0) or 0)
        if emb_dim <= 0:
            raise ValueError("hidden_size/projection_dim missing from text model config")
        self.dim = emb_dim
        if dim is not None and int(dim) != self.dim:
            raise ValueError(f"text encoder dim mismatch: requested {dim}, model provides {self.dim}")

    def _load_model(self) -> tuple[Any, Any, Path]:
        manifest_path = self.model_path / "manifest.json"
        base = self.model_path / "base"
        finetuned = self.model_path / "finetuned"

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            base = self.model_path / str(manifest.get("base_subdir", "base"))
            finetuned = self.model_path / str(manifest.get("finetuned_subdir", "finetuned"))

        if self._is_checkpoint_dir(finetuned):
            source_dir = finetuned
        elif self._is_checkpoint_dir(base):
            source_dir = base
        elif self._is_checkpoint_dir(self.model_path):
            source_dir = self.model_path
        else:
            raise FileNotFoundError(
                f"text model checkpoint not found under {self.model_path}; "
                "download a vendored text model under models/ (e.g. qimg-text-encoder-bge-m3)"
            )

        with _model_load_quiet_ctx():
            model = AutoModel.from_pretrained(str(source_dir)).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(str(source_dir), use_fast=True)
        return model, tokenizer, source_dir

    @staticmethod
    def _is_checkpoint_dir(path: Path) -> bool:
        return (path / "config.json").exists()

    def _resolve_max_length(self) -> int:
        candidates: list[int] = []
        tok_max = int(getattr(self._tokenizer, "model_max_length", 0) or 0)
        cfg_max = int(getattr(self._model.config, "max_position_embeddings", 0) or 0)
        for val in (tok_max, cfg_max):
            if val > 0 and val <= 32768:
                candidates.append(val)
        if not candidates:
            return 512
        return max(candidates)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(text, mode="query")

    def encode_document(self, text: str) -> np.ndarray:
        return self._encode(text, mode="passage")

    def _encode(self, text: str, mode: str) -> np.ndarray:
        raw = (text or "").strip()
        if not raw:
            return np.zeros((self.dim,), dtype=np.float32)

        normalized = raw
        model_l = self.model.lower()
        if "e5" in model_l:
            if mode == "query" and not normalized.lower().startswith("query:"):
                normalized = f"query: {normalized}"
            if mode != "query" and not normalized.lower().startswith("passage:"):
                normalized = f"passage: {normalized}"
        elif "bge" in model_l and "m3" not in model_l and mode == "query":
            prefix = "Represent this sentence for searching relevant passages: "
            if not normalized.lower().startswith(prefix.lower()):
                normalized = f"{prefix}{normalized}"

        inputs = self._tokenizer(
            [normalized],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = self._model(**inputs)
            emb = self._pool(out, inputs.get("attention_mask"), model_l=model_l)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return emb[0].detach().to("cpu").numpy().astype(np.float32)

    @staticmethod
    def _pool(out: Any, attention_mask: torch.Tensor | None, model_l: str) -> torch.Tensor:
        if hasattr(out, "last_hidden_state") and getattr(out, "last_hidden_state") is not None:
            h = getattr(out, "last_hidden_state")
            if "bge" in model_l:
                # BGE model cards recommend CLS pooling for retrieval embeddings.
                return h[:, 0]
            if attention_mask is None:
                return h.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).expand(h.shape).float()
            return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        if hasattr(out, "pooler_output") and getattr(out, "pooler_output") is not None:
            return getattr(out, "pooler_output")
        if isinstance(out, torch.Tensor):
            return out
        raise TypeError(f"unsupported output for text embedding: {type(out)!r}")
