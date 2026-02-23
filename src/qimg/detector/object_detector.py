from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None  # type: ignore[assignment]
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection  # type: ignore

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


@dataclass(slots=True)
class DetectedObject:
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float] | None = None


class LocalObjectDetector:
    """Local object detector loaded from vendored models/<model>/."""

    def __init__(
        self,
        model: str = "qimg-object-detector-detr-resnet-50",
        device: str = "auto",
        threshold: float = 0.30,
        max_detections: int = 12,
    ):
        self.model = (model or "qimg-object-detector-detr-resnet-50").strip()
        self.device = resolve_device(device)
        self.threshold = float(max(0.0, min(1.0, threshold)))
        self.max_detections = int(max(1, max_detections))
        self._model_manager = get_model_manager()
        self._model_manager.sync_required([self.model])
        self.model_path = self._model_manager.model_dir(self.model)
        self._model, self._processor, self._source_dir = self._load_model()
        self._model.eval()
        self.source = f"object_detector:{self.model}:{self.device}"

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
                f"object detector checkpoint not found under {self.model_path}; "
                "download a vendored model under models/ (e.g. qimg-object-detector-detr-resnet-50)"
            )

        with _model_load_quiet_ctx():
            model = AutoModelForObjectDetection.from_pretrained(str(source_dir)).to(self.device)
            processor = AutoImageProcessor.from_pretrained(str(source_dir))
        return model, processor, source_dir

    @staticmethod
    def _is_checkpoint_dir(path: Path) -> bool:
        return (path / "config.json").exists()

    def detect(self, path: Path) -> list[DetectedObject]:
        if Image is None:
            raise RuntimeError("Pillow is required for object detection")

        with Image.open(path) as img:
            image = img.convert("RGB")
            target_h, target_w = image.size[1], image.size[0]

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([[target_h, target_w]], device=self.device)
        processed = self._processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )
        if not processed:
            return []

        result = processed[0]
        scores = result.get("scores")
        labels = result.get("labels")
        boxes = result.get("boxes")
        if scores is None or labels is None:
            return []

        id2label = getattr(self._model.config, "id2label", {}) or {}
        by_label: dict[str, DetectedObject] = {}
        for idx in range(int(scores.shape[0])):  # type: ignore[attr-defined]
            score = float(scores[idx].detach().to("cpu").item())
            label_id = int(labels[idx].detach().to("cpu").item())
            label_raw = id2label.get(label_id, str(label_id))
            label = str(label_raw).strip().lower()
            if not label:
                continue

            bbox_xyxy: tuple[float, float, float, float] | None = None
            if boxes is not None:
                b = boxes[idx].detach().to("cpu").tolist()
                if isinstance(b, list) and len(b) == 4:
                    bbox_xyxy = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

            prev = by_label.get(label)
            if prev is None or score > prev.confidence:
                by_label[label] = DetectedObject(label=label, confidence=score, bbox_xyxy=bbox_xyxy)

        ranked = sorted(by_label.values(), key=lambda x: x.confidence, reverse=True)
        return ranked[: self.max_detections]
