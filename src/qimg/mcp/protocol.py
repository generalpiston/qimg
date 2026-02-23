from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qimg.service import QimgService, SearchFilters


@dataclass(slots=True)
class MCPResponse:
    ok: bool
    result: Any | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {"ok": self.ok}
        if self.ok:
            out["result"] = self.result
        else:
            out["error"] = self.error or "unknown error"
        return out


def _filters_from_args(args: dict[str, Any]) -> SearchFilters:
    return SearchFilters(
        num=int(args.get("num", 10)),
        collection=args.get("collection"),
        path_prefix=args.get("path_prefix"),
        after=args.get("after"),
        before=args.get("before"),
        min_width=args.get("min_width"),
        min_height=args.get("min_height"),
        min_score=float(args.get("min_score", 0.0)),
        all_results=bool(args.get("all", False)),
    )


class MCPProtocol:
    def __init__(self, service: QimgService):
        self.service = service

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        tool = payload.get("tool") or payload.get("method")
        args = payload.get("args") or payload.get("params") or {}

        try:
            if tool == "qimg_search":
                query = str(args.get("query", ""))
                result = self.service.search(query, _filters_from_args(args))
                return MCPResponse(ok=True, result=result).as_dict()

            if tool == "qimg_vector_search":
                query = str(args.get("query", ""))
                result = self.service.vsearch(query, _filters_from_args(args))
                return MCPResponse(ok=True, result=result).as_dict()

            if tool == "qimg_deep_search":
                query = str(args.get("query", ""))
                result = self.service.query(query, _filters_from_args(args))
                return MCPResponse(ok=True, result=result).as_dict()

            if tool == "qimg_get":
                identifier = str(args.get("id") or args.get("path") or "")
                result = self.service.get(identifier)
                if result is None:
                    suggestions = self._suggest(identifier)
                    return MCPResponse(ok=False, error=f"not found; suggestions={suggestions}").as_dict()
                return MCPResponse(ok=True, result=result).as_dict()

            if tool == "qimg_multi_get":
                identifiers = args.get("ids") or args.get("paths")
                glob_pattern = args.get("glob")
                if isinstance(identifiers, str):
                    identifiers = [identifiers]
                if identifiers is None:
                    identifiers = []
                result = self.service.multi_get(identifiers=identifiers, glob_pattern=glob_pattern)
                return MCPResponse(ok=True, result=result).as_dict()

            if tool == "qimg_status":
                return MCPResponse(ok=True, result=self.service.status()).as_dict()

            if tool == "qimg_facts_extract":
                ids = args.get("ids") or args.get("assets") or []
                if isinstance(ids, str):
                    ids = [ids]
                result = self.service.facts_extract(
                    extract_caption=bool(args.get("caption", False)),
                    extract_tags=bool(args.get("tags", False)),
                    extract_objects=bool(args.get("objects", False)),
                    extract_ocr=bool(args.get("ocr", False)),
                    identifiers=ids,
                )
                return MCPResponse(ok=True, result=result).as_dict()

            if tool == "qimg_facts_get":
                identifier = str(args.get("id") or args.get("path") or "")
                result = self.service.facts_ls(identifier)
                return MCPResponse(ok=True, result=result).as_dict()

            return MCPResponse(ok=False, error=f"unknown tool: {tool}").as_dict()
        except Exception as exc:  # pragma: no cover
            return MCPResponse(ok=False, error=str(exc)).as_dict()

    def _suggest(self, text: str, limit: int = 5) -> list[str]:
        if not text:
            return []
        needle = text.strip("#").lower()
        like = f"%{needle}%"
        with self.service.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, abs_path
                FROM assets
                WHERE LOWER(abs_path) LIKE ? OR LOWER(rel_path) LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (like, like, limit),
            ).fetchall()
        return [f"{r['id']} {r['abs_path']}" for r in rows]
