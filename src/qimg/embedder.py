from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Sequence

import numpy as np
try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None  # type: ignore[assignment]


@dataclass(slots=True)
class EmbedderConfig:
    model: str = "qimg-encoder-finetuned-clip-vit-large-patch14"
    text_model: str = "qimg-text-encoder-bge-m3"
    device: str = "auto"


class _FallbackEmbedder:
    def __init__(self, dim: int = 192):
        self.dim = dim
        self.device = "cpu"
        self.model_id = f"fallback-hist-{dim}"
        self.text_model_id = self.model_id

    def embed_image(self, path: Path) -> np.ndarray:
        if Image is None:
            return self._embed_file_bytes(path)

        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize((96, 96))
                arr = np.asarray(img, dtype=np.float32) / 255.0
        except Exception:
            return self._embed_file_bytes(path)

        channels = []
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c], bins=64, range=(0.0, 1.0), density=False)
            channels.append(hist.astype(np.float32))

        vec = np.concatenate(channels)
        if vec.shape[0] < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.shape[0]))
        elif vec.shape[0] > self.dim:
            vec = vec[: self.dim]
        return _normalize(vec)

    def _embed_file_bytes(self, path: Path) -> np.ndarray:
        vec = np.zeros((self.dim,), dtype=np.float32)
        try:
            data = path.read_bytes()
        except Exception:
            return vec
        if not data:
            return vec
        chunk = 64
        for i in range(0, len(data), chunk):
            part = data[i : i + chunk]
            digest = hashlib.sha256(part).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            vec[idx] += 1.0
        return _normalize(vec)

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros((self.dim,), dtype=np.float32)
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            i = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            vec[i] += sign
        return _normalize(vec)


class _VendoredLocalEmbedder:
    def __init__(self, model: str, text_model: str, device: str):
        from qimg.encoder import LocalImageEncoder, LocalTextEncoder

        if text_model.strip() == model.strip():
            raise ValueError("embed.text_model must be different from embed.model")
        self._image_encoder = LocalImageEncoder(model=model, dim=None, device=device)
        self._text_encoder = LocalTextEncoder(model=text_model, dim=None, device=device)
        self.device = str(self._image_encoder.device)
        self.dim = int(self._image_encoder.dim)
        self.model_id = f"vendored-local:{model}:{self.device}:d{self.dim}"
        self.text_model_id = f"vendored-local-text:{text_model}:{self.device}:d{self._text_encoder.dim}"

    def embed_image(self, path: Path) -> np.ndarray:
        return self._image_encoder.encode(path)

    def embed_image_query(self, text: str) -> np.ndarray:
        return self._image_encoder.encode_text(text)

    def embed_text_query(self, text: str) -> np.ndarray:
        return self._text_encoder.encode_query(text)

    def embed_text_document(self, text: str) -> np.ndarray:
        return self._text_encoder.encode_document(text)

    def embed_text(self, text: str) -> np.ndarray:
        # Chunk long text and average encoded passage vectors for better OCR sentence coverage.
        words = [w for w in re.findall(r"\S+", text.strip()) if w]
        if not words:
            return np.zeros((self._text_encoder.dim,), dtype=np.float32)
        if len(words) <= 64:
            return self._text_encoder.encode_document(" ".join(words))

        chunks: list[str] = []
        step = 48
        width = 64
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + width])
            if chunk:
                chunks.append(chunk)
            if i + width >= len(words):
                break

        vecs = [self._text_encoder.encode_document(c) for c in chunks]
        avg = np.mean(np.vstack([v.reshape(1, -1) for v in vecs]), axis=0)
        norm = float(np.linalg.norm(avg))
        if norm <= 1e-12:
            return avg.astype(np.float32)
        return (avg / norm).astype(np.float32)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


class Embedder:
    def __init__(
        self,
        model: str = "qimg-encoder-finetuned-clip-vit-large-patch14",
        text_model: str = "qimg-text-encoder-bge-m3",
        device: str = "auto",
    ):
        self._impl: _FallbackEmbedder | _VendoredLocalEmbedder
        name = (model or "").strip().lower()
        if name in {"fallback-clip", "fallback-hist"}:
            if os.environ.get("QIMG_ALLOW_FALLBACK", "").strip() != "1":
                raise RuntimeError(
                    "fallback embedder is disabled; use a vendored model under models/ "
                    "(e.g. qimg-encoder-finetuned-clip-vit-large-patch14) "
                    "or set QIMG_ALLOW_FALLBACK=1 to override."
                )
            self._impl = _FallbackEmbedder()
            return

        self._impl = _VendoredLocalEmbedder(model=model, text_model=text_model, device=device)

    @property
    def model_id(self) -> str:
        return self._impl.model_id

    @property
    def text_model_id(self) -> str:
        return str(getattr(self._impl, "text_model_id", self._impl.model_id))

    @property
    def dim(self) -> int:
        return int(self._impl.dim)

    @property
    def device(self) -> str:
        return str(self._impl.device)

    def embed_image(self, path: Path) -> np.ndarray:
        return self._impl.embed_image(path)

    def embed_images(self, paths: Sequence[Path]) -> list[np.ndarray]:
        return [self.embed_image(p) for p in paths]

    def embed_text(self, text: str) -> np.ndarray:
        return self._impl.embed_text(text)

    def embed_image_query(self, text: str) -> np.ndarray:
        if hasattr(self._impl, "embed_image_query"):
            return self._impl.embed_image_query(text)
        return self._impl.embed_text(text)

    def embed_text_query(self, text: str) -> np.ndarray:
        if hasattr(self._impl, "embed_text_query"):
            return self._impl.embed_text_query(text)
        return self._impl.embed_text(text)

    def embed_text_document(self, text: str) -> np.ndarray:
        if hasattr(self._impl, "embed_text_document"):
            return self._impl.embed_text_document(text)
        return self._impl.embed_text(text)
