from pathlib import Path

from qimg.model_manager import (
    ALL_SPECS,
    DownloadResult,
    ModelManager,
    ModelSpec,
)


def test_resolve_specs_defaults_to_all_models() -> None:
    mgr = ModelManager()
    specs = mgr.resolve_specs(download_image=False, download_text=False, download_object=False)
    assert specs == list(ALL_SPECS)


def test_resolve_specs_honors_flags() -> None:
    mgr = ModelManager()
    specs = mgr.resolve_specs(download_image=True, download_text=False, download_object=True)
    assert [s.kind for s in specs] == ["image", "object"]


def test_sync_required_downloads_missing_model(tmp_path: Path, monkeypatch) -> None:
    mgr = ModelManager(models_root=tmp_path / "models")
    calls: list[ModelSpec] = []

    def _stub(spec: ModelSpec, from_cache: bool = False) -> DownloadResult:
        calls.append(spec)
        base = mgr.model_dir(spec.model_name) / "base"
        base.mkdir(parents=True, exist_ok=True)
        (base / "config.json").write_text("{}")
        return DownloadResult(
            kind=spec.kind,
            model_name=spec.model_name,
            base_hf_model=spec.base_hf_model,
            revision=spec.revision,
            base_dir=str(base),
        )

    monkeypatch.setattr(mgr, "download_spec", _stub)
    results = mgr.sync_required([ALL_SPECS[0].model_name])

    assert [c.kind for c in calls] == [ALL_SPECS[0].kind]
    assert len(results) == 1
    assert mgr.has_checkpoint(ALL_SPECS[0].model_name)
