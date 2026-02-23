from __future__ import annotations

import argparse

from qimg.model_manager import (
    ALL_SPECS,
    IMAGE_SPEC,
    OBJECT_SPEC,
    TEXT_SPEC,
    DownloadResult,
    ModelManager,
    ModelSpec,
    get_model_manager,
    result_to_dict,
)


def resolve_specs(download_image: bool, download_text: bool, download_object: bool) -> list[ModelSpec]:
    return get_model_manager().resolve_specs(download_image, download_text, download_object)


def download_model_spec(spec: ModelSpec, from_cache: bool = False) -> DownloadResult:
    return get_model_manager().download_spec(spec, from_cache=from_cache)


def download_selected(
    download_image: bool = False,
    download_text: bool = False,
    download_object: bool = False,
    from_cache: bool = False,
) -> list[DownloadResult]:
    return get_model_manager().download_selected(
        download_image=download_image,
        download_text=download_text,
        download_object=download_object,
        from_cache=from_cache,
    )


def _cli_single(default_spec: ModelSpec) -> None:
    parser = argparse.ArgumentParser(description=f"Download base HF {default_spec.kind} model for qimg")
    parser.add_argument("--model-name", default=default_spec.model_name)
    parser.add_argument("--base-hf-model", default=default_spec.base_hf_model)
    parser.add_argument("--revision", default=default_spec.revision)
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Copy latest snapshot from local ~/.cache/huggingface/hub instead of downloading",
    )
    args = parser.parse_args()
    spec = ModelSpec(
        kind=default_spec.kind,
        model_name=str(args.model_name),
        base_hf_model=str(args.base_hf_model),
        revision=str(args.revision),
        task=default_spec.task,
        license_name=default_spec.license_name,
        readme_title=default_spec.readme_title,
    )
    manager = ModelManager()
    result = manager.download_spec(spec, from_cache=bool(args.from_cache))
    print(result.base_dir)


def cli_download_image_model() -> None:
    _cli_single(IMAGE_SPEC)


def cli_download_text_model() -> None:
    _cli_single(TEXT_SPEC)


def cli_download_object_model() -> None:
    _cli_single(OBJECT_SPEC)


__all__ = [
    "ALL_SPECS",
    "IMAGE_SPEC",
    "TEXT_SPEC",
    "OBJECT_SPEC",
    "DownloadResult",
    "ModelSpec",
    "ModelManager",
    "download_model_spec",
    "download_selected",
    "resolve_specs",
    "result_to_dict",
    "cli_download_image_model",
    "cli_download_text_model",
    "cli_download_object_model",
]
