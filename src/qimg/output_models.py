from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover - fallback when pydantic is unavailable.

    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **data: Any):
            annotations = getattr(self.__class__, "__annotations__", {})
            for field in annotations:
                if field in data:
                    value = data[field]
                elif hasattr(self.__class__, field):
                    value = deepcopy(getattr(self.__class__, field))
                else:
                    value = None
                setattr(self, field, value)

        def model_dump(self) -> dict[str, Any]:
            def _dump(value: Any) -> Any:
                if isinstance(value, BaseModel):
                    return value.model_dump()
                if isinstance(value, list):
                    return [_dump(v) for v in value]
                if isinstance(value, dict):
                    return {k: _dump(v) for k, v in value.items()}
                return value

            return {name: _dump(getattr(self, name)) for name in getattr(self.__class__, "__annotations__", {})}


class ContextOutput(BaseModel):
    virtual_path: str
    text: str


class FactLabelOutput(BaseModel):
    label: str
    conf: float | None = None


class FactsOutput(BaseModel):
    caption: str | None = None
    tags: list[FactLabelOutput] = []
    objects: list[FactLabelOutput] = []
    ocr: str | None = None
    exif: dict[str, Any] = {}
    derived: dict[str, Any] = {}


class MetadataOutput(BaseModel):
    collection: str
    rel_path: str
    abs_path: str
    width: int | None = None
    height: int | None = None
    size: int
    mtime: float
    date: str | None = None
    camera: str | None = None


class ScoresOutput(BaseModel):
    overall: float
    rrf: float | None = None
    lexical_total: float | None = None
    lexical_namefolder: float | None = None
    lexical_context: float | None = None
    lexical_facts: float | None = None
    lexical_ocr: float | None = None
    vector: float | None = None
    vector_image: float | None = None
    vector_ocr: float | None = None
    rerank: float | None = None
    contributions: list[str] = []


class SearchResultOutput(BaseModel):
    id: str
    path: str
    rel_path: str
    collection: str
    metadata: MetadataOutput
    contexts: list[ContextOutput] = []
    facts: FactsOutput
    scores: ScoresOutput
