This folder contains vendored runtime model artifacts managed by `qimg`.

Runtime model directories:
- `qimg-encoder-finetuned-clip-vit-large-patch14/`: image-text encoder used for image embeddings and image-text retrieval.
- `qimg-text-encoder-bge-m3/`: text embedding model used for OCR/query semantic retrieval.
- `qimg-object-detector-detr-resnet-50/`: object detector used by `qimg facts extract --objects`.

Expected per-model layout:
- `base/`: base checkpoint files.
- `finetuned/` (optional): local fine-tuned checkpoint files; preferred at runtime when present.
- `manifest.json`: model metadata (`base_model`, `revision`, task, and subdir pointers).
- `README.md`: local notes for the vendored model.

How models are managed:
- Manual sync: `uv run --extra encoder qimg models download`
- From local HF cache only: `uv run --extra encoder qimg models download --from-cache`
- Automatic sync: missing built-in models are downloaded on first use by runtime commands.

Model code lives under `src/qimg/model_manager.py`, `src/qimg/encoder/`, and `src/qimg/detector/`.
Set `QIMG_MODELS_DIR` to override the default models root.
