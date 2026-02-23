from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any


@dataclass(slots=True, frozen=True)
class ModelSpec:
    kind: str
    model_name: str
    base_hf_model: str
    revision: str
    task: str
    license_name: str
    readme_title: str


@dataclass(slots=True)
class DownloadResult:
    kind: str
    model_name: str
    base_hf_model: str
    revision: str
    base_dir: str


IMAGE_SPEC = ModelSpec(
    kind="image",
    model_name="qimg-encoder-finetuned-clip-vit-large-patch14",
    base_hf_model="openai/clip-vit-large-patch14",
    revision="main",
    task="image_text_retrieval",
    license_name="mit",
    readme_title="image-text retrieval",
)

TEXT_SPEC = ModelSpec(
    kind="text",
    model_name="qimg-text-encoder-bge-m3",
    base_hf_model="BAAI/bge-m3",
    revision="main",
    task="text_embedding",
    license_name="unknown",
    readme_title="text embedding (OCR/query semantic retrieval)",
)

OBJECT_SPEC = ModelSpec(
    kind="object",
    model_name="qimg-object-detector-detr-resnet-50",
    base_hf_model="facebook/detr-resnet-50",
    revision="no_timm",
    task="object_detection",
    license_name="unknown",
    readme_title="object detection for qimg facts extraction",
)

ALL_SPECS: tuple[ModelSpec, ...] = (IMAGE_SPEC, TEXT_SPEC, OBJECT_SPEC)


def result_to_dict(result: DownloadResult) -> dict[str, Any]:
    return {
        "kind": result.kind,
        "model_name": result.model_name,
        "base_hf_model": result.base_hf_model,
        "revision": result.revision,
        "base_dir": result.base_dir,
    }


def _cache_dir_for_repo(repo_id: str) -> Path:
    slug = repo_id.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{slug}"


def _default_models_root() -> Path:
    env = os.environ.get("QIMG_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return (parent / "models").resolve()
    return (Path.cwd() / "models").resolve()


def _copy_snapshot(repo_id: str, base_dir: Path, preferred_revision: str | None) -> str:
    cache_dir = _cache_dir_for_repo(repo_id)
    snapshots = cache_dir / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"no local HF cache snapshots found for {repo_id}: {snapshots}")

    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no snapshot directories found for {repo_id}: {snapshots}")

    chosen = None
    if preferred_revision:
        preferred = snapshots / preferred_revision
        if preferred.exists() and preferred.is_dir():
            chosen = preferred
    if chosen is None:
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)

    base_dir.mkdir(parents=True, exist_ok=True)
    for item in chosen.iterdir():
        dst = base_dir / item.name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if item.is_dir():
            shutil.copytree(item, dst, symlinks=False)
        else:
            shutil.copy2(item.resolve(), dst)
    return chosen.name


def _is_checkpoint_dir(path: Path) -> bool:
    return (path / "config.json").exists()


class ModelManager:
    def __init__(self, models_root: Path | None = None):
        self._models_root = (models_root or _default_models_root()).expanduser().resolve()
        self._spec_by_name: dict[str, ModelSpec] = {spec.model_name: spec for spec in ALL_SPECS}

    @property
    def models_root(self) -> Path:
        return self._models_root

    def model_dir(self, model_name: str) -> Path:
        return self._models_root / model_name

    def resolve_specs(self, download_image: bool, download_text: bool, download_object: bool) -> list[ModelSpec]:
        if not any([download_image, download_text, download_object]):
            return list(ALL_SPECS)
        selected: list[ModelSpec] = []
        if download_image:
            selected.append(IMAGE_SPEC)
        if download_text:
            selected.append(TEXT_SPEC)
        if download_object:
            selected.append(OBJECT_SPEC)
        return selected

    def has_checkpoint(self, model_name: str) -> bool:
        model_path = self.model_dir(model_name)
        manifest_path = model_path / "manifest.json"
        base = model_path / "base"
        finetuned = model_path / "finetuned"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                base = model_path / str(manifest.get("base_subdir", "base"))
                finetuned = model_path / str(manifest.get("finetuned_subdir", "finetuned"))
            except Exception:
                pass
        return _is_checkpoint_dir(finetuned) or _is_checkpoint_dir(base) or _is_checkpoint_dir(model_path)

    def _write_manifest_and_readme(self, spec: ModelSpec, model_dir: Path, used_revision: str) -> None:
        manifest = {
            "model": spec.model_name,
            "base_model": spec.base_hf_model,
            "revision": used_revision,
            "license": spec.license_name,
            "task": spec.task,
            "base_subdir": "base",
            "finetuned_subdir": "finetuned",
        }
        (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        readme = (
            f"Vendored model assets for `{spec.model_name}`.\n\n"
            f"- base checkpoint: `{spec.base_hf_model}`\n"
            f"- task: {spec.readme_title}\n"
        )
        (model_dir / "README.md").write_text(readme)

    def download_spec(self, spec: ModelSpec, from_cache: bool = False) -> DownloadResult:
        model_dir = self.model_dir(spec.model_name)
        base_dir = model_dir / "base"
        model_dir.mkdir(parents=True, exist_ok=True)

        used_revision = spec.revision
        if from_cache:
            used_revision = _copy_snapshot(spec.base_hf_model, base_dir, preferred_revision=spec.revision)
        else:
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=spec.base_hf_model,
                    revision=spec.revision,
                    local_dir=str(base_dir),
                )
            except Exception:
                used_revision = _copy_snapshot(spec.base_hf_model, base_dir, preferred_revision=spec.revision)

        self._write_manifest_and_readme(spec, model_dir, used_revision=used_revision)
        return DownloadResult(
            kind=spec.kind,
            model_name=spec.model_name,
            base_hf_model=spec.base_hf_model,
            revision=used_revision,
            base_dir=str(base_dir),
        )

    def download_selected(
        self,
        download_image: bool = False,
        download_text: bool = False,
        download_object: bool = False,
        from_cache: bool = False,
    ) -> list[DownloadResult]:
        specs = self.resolve_specs(download_image, download_text, download_object)
        return [self.download_spec(spec, from_cache=from_cache) for spec in specs]

    def sync(
        self,
        download_image: bool = False,
        download_text: bool = False,
        download_object: bool = False,
        from_cache: bool = False,
        force: bool = False,
    ) -> list[DownloadResult]:
        specs = self.resolve_specs(download_image, download_text, download_object)
        out: list[DownloadResult] = []
        for spec in specs:
            if force or not self.has_checkpoint(spec.model_name):
                out.append(self.download_spec(spec, from_cache=from_cache))
        return out

    def ensure(self, model_name: str, from_cache: bool = False) -> DownloadResult | None:
        if self.has_checkpoint(model_name):
            return None
        spec = self._spec_by_name.get(model_name)
        if spec is None:
            raise FileNotFoundError(
                f"model checkpoint not found under {self.model_dir(model_name)} and no built-in download spec exists for '{model_name}'"
            )
        return self.download_spec(spec, from_cache=from_cache)

    def sync_required(self, model_names: list[str], from_cache: bool = False) -> list[DownloadResult]:
        out: list[DownloadResult] = []
        for model_name in model_names:
            result = self.ensure(model_name, from_cache=from_cache)
            if result is not None:
                out.append(result)
        return out


_DEFAULT_MANAGER: ModelManager | None = None


def get_model_manager() -> ModelManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = ModelManager()
    return _DEFAULT_MANAGER
