from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import fnmatch
from pathlib import Path
import sqlite3
from typing import Any
import numpy as np

from qimg import contexts as contexts_mod
from qimg.collections import add_collection, get_collection_by_name, list_collections, remove_collection, rename_collection
from qimg.config import AppConfig
from qimg.db import Database, decode_vector
from qimg.embedder import Embedder
from qimg.facts import extract_facts, list_all_facts, list_facts, remove_facts_by_source, summarize_facts
from qimg.fts import refresh_lexical_for_all, tokenize
from qimg.ids import parse_short_id, short_id_from_int
from qimg.indexer import cleanup_orphans, sync_collections
from qimg.output_models import (
    ContextOutput,
    FactLabelOutput,
    FactsOutput,
    MetadataOutput,
    ScoresOutput,
    SearchResultOutput,
)
from qimg.query.expand import expand_query_fallback
from qimg.rank.normalize import bm25_to_unit
from qimg.rank.rerank import CandidateText, OverlapReranker, blend_retrieval_and_rerank
from qimg.rank.rrf import fuse_rankings
from qimg.vectors import VectorIndex, rebuild_index_from_db, upsert_vector_row


@dataclass(slots=True)
class SearchFilters:
    num: int = 10
    collection: str | None = None
    path_prefix: str | None = None
    after: str | None = None
    before: str | None = None
    min_width: int | None = None
    min_height: int | None = None
    min_score: float = 0.0
    all_results: bool = False


@dataclass(slots=True)
class LexicalScore:
    asset_id: str
    total: float
    namefolder: float
    context: float
    facts: float


def _parse_date(date_value: str | None) -> float | None:
    if not date_value:
        return None
    try:
        dt = datetime.fromisoformat(date_value)
    except ValueError:
        return None
    return dt.timestamp()


def _apply_filters_sql(filters: SearchFilters) -> tuple[str, list[Any]]:
    clauses = ["a.is_deleted = 0"]
    args: list[Any] = []

    if filters.collection:
        clauses.append("c.name = ?")
        args.append(filters.collection)
    if filters.path_prefix:
        clauses.append("a.rel_path LIKE ?")
        args.append(f"{filters.path_prefix}%")
    if filters.min_width is not None:
        clauses.append("COALESCE(a.width, 0) >= ?")
        args.append(filters.min_width)
    if filters.min_height is not None:
        clauses.append("COALESCE(a.height, 0) >= ?")
        args.append(filters.min_height)

    after_ts = _parse_date(filters.after)
    if after_ts is not None:
        clauses.append("a.mtime >= ?")
        args.append(after_ts)
    before_ts = _parse_date(filters.before)
    if before_ts is not None:
        clauses.append("a.mtime <= ?")
        args.append(before_ts)

    return " AND ".join(clauses), args


def _query_channel_bias(query: str) -> tuple[float, float]:
    tokens = [t.lower() for t in query.replace("/", " ").split()]
    broad_cues = {"style", "scene", "mood", "aesthetic", "vibe", "trip", "family", "album", "collection"}
    specific_cues = {"object", "objects", "tag", "tags", "ocr", "text", "caption", "logo"}

    if any(t in specific_cues for t in tokens):
        return 0.95, 1.25
    if any(t in broad_cues for t in tokens) or len(tokens) <= 2:
        return 1.20, 0.95
    return 1.0, 1.0


def _vector_channel_weights(query: str) -> tuple[float, float]:
    q = query.lower()
    tokens = q.replace("/", " ").replace("-", " ").split()
    text_cues = {
        "ocr",
        "text",
        "invoice",
        "receipt",
        "serial",
        "order",
        "id",
        "code",
        "email",
        "phone",
        "address",
        "error",
        "stacktrace",
        "log",
    }
    has_digits = sum(ch.isdigit() for ch in q) >= 2
    texty = has_digits or any(t in text_cues for t in tokens) or "@" in q or "http" in q
    if texty:
        return 0.90, 1.35
    return 1.0, 1.10


def _contributions(scores: ScoresOutput) -> list[str]:
    out: list[str] = []
    if scores.lexical_namefolder:
        out.append("lexical:namefolder")
    if scores.lexical_context:
        out.append("lexical:context")
    if scores.lexical_facts:
        out.append("lexical:facts")
    if scores.lexical_ocr:
        out.append("lexical:ocr")
    if scores.vector_image:
        out.append("vector:image")
    if scores.vector_ocr:
        out.append("vector:ocr")
    if scores.vector:
        out.append("vector")
    if scores.rerank is not None:
        out.append("rerank")
    return out


def _normalize_ranked_rrf(ranked: list[tuple[str, float]]) -> dict[str, float]:
    if not ranked:
        return {}
    max_score = max(float(score) for _, score in ranked)
    if max_score <= 1e-12:
        return {str(asset_id): 0.0 for asset_id, _ in ranked}
    return {str(asset_id): float(score) / max_score for asset_id, score in ranked}


class QimgService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.db = Database(config.db_path)
        self.db.initialize()
        self._embedder_instance: Embedder | None = None
        self._vector_indexes: dict[int, VectorIndex] = {}

    def _embedder(self) -> Embedder:
        if self._embedder_instance is None:
            self._embedder_instance = Embedder(
                model=self.config.embed.model,
                text_model=self.config.embed.text_model,
                device=self.config.embed.device,
            )
        return self._embedder_instance

    def _vector_index(self, dim: int) -> VectorIndex:
        idx = self._vector_indexes.get(dim)
        if idx is None:
            idx = VectorIndex(dim=dim, index_path=self.config.vector_index_path, fallback_path=self.config.vector_fallback_path)
            self._vector_indexes[dim] = idx
        return idx

    def collection_add(self, root_path: str, name: str, mask: str) -> dict[str, Any]:
        with self.db.connect() as conn:
            cid = add_collection(conn, name=name, root_path=root_path, mask=mask)
        return {"id": cid, "name": name, "root_path": root_path, "mask": mask}

    def collection_list(self) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = list_collections(conn)
        return [
            {
                "id": r.id,
                "name": r.name,
                "root_path": r.root_path,
                "mask": r.mask,
                "created_at": r.created_at,
            }
            for r in rows
        ]

    def collection_remove(self, name: str) -> None:
        with self.db.connect() as conn:
            remove_collection(conn, name)

    def collection_rename(self, old_name: str, new_name: str) -> None:
        with self.db.connect() as conn:
            rename_collection(conn, old_name, new_name)

    def context_add(self, virtual_path: str, text: str) -> dict[str, Any]:
        with self.db.connect() as conn:
            ctx_id = contexts_mod.add_context(conn, virtual_path, text)
            contexts_mod.materialize_context_map(conn)
            refresh_lexical_for_all(conn)
        return {"id": ctx_id, "virtual_path": virtual_path, "text": text}

    def context_list(self, prefix: str | None = None) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = contexts_mod.list_contexts(conn, prefix=prefix)
        return [r.__dict__ for r in rows]

    def context_rm(self, virtual_path: str) -> None:
        with self.db.connect() as conn:
            contexts_mod.remove_context(conn, virtual_path)
            contexts_mod.materialize_context_map(conn)
            refresh_lexical_for_all(conn)

    def update(self) -> dict[str, Any]:
        with self.db.connect() as conn:
            stats, changed_ids = sync_collections(conn, compute_sha256=self.config.compute_sha256)
        return {
            "added": stats.added,
            "changed": stats.changed,
            "deleted": stats.deleted,
            "scanned": stats.scanned,
            "changed_ids": len(changed_ids),
        }

    def facts_extract(
        self,
        extract_caption: bool,
        extract_tags: bool,
        extract_objects: bool,
        extract_ocr: bool,
        identifiers: list[str] | None = None,
    ) -> dict[str, Any]:
        # If no extraction channel is explicitly selected, run all channels.
        if not any([extract_caption, extract_tags, extract_objects, extract_ocr]):
            extract_caption = True
            extract_tags = True
            extract_objects = True
            extract_ocr = True

        resolved: list[str] = []
        with self.db.connect() as conn:
            if identifiers:
                for item in identifiers:
                    asset_id = self._resolve_identifier(conn, item)
                    if asset_id and asset_id not in resolved:
                        resolved.append(asset_id)

            stats = extract_facts(
                conn,
                asset_ids=resolved or None,
                extract_caption=extract_caption,
                extract_tags=extract_tags,
                extract_objects=extract_objects,
                extract_ocr=extract_ocr,
            )
            refresh_lexical_for_all(conn)

        return {
            "processed": stats.processed,
            "captions": stats.captions,
            "tags": stats.tags,
            "objects": stats.objects,
            "ocr": stats.ocr,
            "errors": stats.errors,
            "object_runtime_available": stats.object_runtime_available,
            "object_skipped_unavailable": stats.object_skipped_unavailable,
            "ocr_runtime_available": stats.ocr_runtime_available,
            "ocr_skipped_unavailable": stats.ocr_skipped_unavailable,
            "scoped_assets": len(resolved),
        }

    def facts_ls(self, identifier: str | None = None) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            if identifier is None or not identifier.strip():
                return list_all_facts(conn)
            asset_id = self._resolve_identifier(conn, identifier)
            if asset_id is None:
                return []
            return list_facts(conn, asset_id)

    def facts_rm(self, source: str, identifier: str | None = None) -> dict[str, Any]:
        with self.db.connect() as conn:
            asset_id: str | None = None
            if identifier:
                asset_id = self._resolve_identifier(conn, identifier)
            removed = remove_facts_by_source(conn, source=source, asset_id=asset_id)
            refresh_lexical_for_all(conn)
        return {"removed": removed, "source": source, "asset_id": asset_id}

    def embed(self, include_images: bool = True, include_ocr_text: bool = True) -> dict[str, Any]:
        embedder = self._embedder()
        index = self._vector_index(embedder.dim)

        embedded_images = 0
        embedded_ocr = 0
        errors = 0
        skipped_missing = 0

        with self.db.connect() as conn:
            if include_images:
                image_rows = conn.execute(
                    """
                    SELECT a.id, a.abs_path, a.updated_at, v.updated_at AS v_updated
                    FROM assets a
                    LEFT JOIN vectors v ON v.asset_id = a.id AND v.model_id = ?
                    WHERE a.is_deleted = 0 AND (
                      v.asset_id IS NULL OR
                      a.updated_at > v.updated_at
                    )
                    ORDER BY a.id
                    """,
                    (embedder.model_id,),
                ).fetchall()

                for row in image_rows:
                    asset_id = str(row["id"])
                    path = Path(str(row["abs_path"]))
                    if not path.exists():
                        conn.execute("UPDATE assets SET is_deleted = 1 WHERE id = ?", (asset_id,))
                        skipped_missing += 1
                        continue
                    try:
                        vec = embedder.embed_image(path)
                    except Exception:
                        errors += 1
                        continue
                    upsert_vector_row(conn, asset_id, embedder.model_id, vec)
                    index.upsert(asset_id, vec)
                    embedded_images += 1

            if include_ocr_text:
                ocr_model_id = f"{embedder.text_model_id}:ocr"
                ocr_rows = conn.execute(
                    """
                    SELECT a.id, a.updated_at, v.updated_at AS v_updated
                    FROM assets a
                    LEFT JOIN vectors v ON v.asset_id = a.id AND v.model_id = ?
                    WHERE a.is_deleted = 0 AND (
                      v.asset_id IS NULL OR
                      a.updated_at > v.updated_at
                    )
                    ORDER BY a.id
                    """,
                    (ocr_model_id,),
                ).fetchall()

                for row in ocr_rows:
                    asset_id = str(row["id"])
                    ocr_text_rows = conn.execute(
                        """
                        SELECT value_text
                        FROM facts
                        WHERE asset_id = ? AND fact_type = 'ocr' AND value_text IS NOT NULL
                        ORDER BY updated_at DESC, id DESC
                        """,
                        (asset_id,),
                    ).fetchall()
                    if not ocr_text_rows:
                        continue
                    text = "\n".join(str(r["value_text"]).strip() for r in ocr_text_rows if str(r["value_text"]).strip())
                    if not text:
                        continue
                    try:
                        vec = embedder.embed_text_document(text)
                    except Exception:
                        errors += 1
                        continue
                    upsert_vector_row(conn, asset_id, ocr_model_id, vec)
                    embedded_ocr += 1

            stale = conn.execute("SELECT id FROM assets WHERE is_deleted = 1").fetchall()
            stale_ids = [str(x["id"]) for x in stale]
            if stale_ids:
                placeholders = ",".join("?" for _ in stale_ids)
                conn.execute(f"DELETE FROM vectors WHERE asset_id IN ({placeholders})", stale_ids)
                index.remove_asset_ids(stale_ids)

        if include_images:
            index.persist()
        return {
            "embedded": embedded_images + embedded_ocr,
            "embedded_images": embedded_images,
            "embedded_ocr_text": embedded_ocr,
            "errors": errors,
            "skipped_missing": skipped_missing,
            "model_id": embedder.model_id,
            "dim": embedder.dim,
            "include_images": include_images,
            "include_ocr_text": include_ocr_text,
        }

    def _allowed_ids(self, conn: sqlite3.Connection, filters: SearchFilters) -> set[str]:
        sql_filter, args = _apply_filters_sql(filters)
        rows = conn.execute(
            f"""
            SELECT a.id
            FROM assets a
            JOIN collections c ON c.id = a.collection_id
            WHERE {sql_filter}
            """,
            args,
        ).fetchall()
        return {str(r["id"]) for r in rows}

    def _lexical_scores(
        self,
        conn: sqlite3.Connection,
        query: str,
        filters: SearchFilters,
        limit: int,
        w_nf: float,
        w_ctx: float,
        w_facts: float,
    ) -> list[LexicalScore]:
        sql_filter, args = _apply_filters_sql(filters)
        fetch_limit = limit if filters.all_results else max(limit * 4, 200)

        rows = conn.execute(
            f"""
            SELECT
              assets_fts.asset_id,
              bm25(assets_fts, 0.0, 1.0, 1.0, 0.0, 0.0) AS bm25_nf,
              bm25(assets_fts, 0.0, 0.0, 0.0, 1.0, 0.0) AS bm25_ctx,
              bm25(assets_fts, 0.0, 0.0, 0.0, 0.0, 1.0) AS bm25_facts
            FROM assets_fts
            JOIN assets a ON a.id = assets_fts.asset_id
            JOIN collections c ON c.id = a.collection_id
            WHERE {sql_filter} AND assets_fts MATCH ?
            ORDER BY bm25(assets_fts)
            LIMIT ?
            """,
            [*args, query, fetch_limit],
        ).fetchall()

        scored: list[LexicalScore] = []
        for row in rows:
            score_nf = bm25_to_unit(float(abs(row["bm25_nf"] or 0.0)))
            score_ctx = bm25_to_unit(float(abs(row["bm25_ctx"] or 0.0)))
            score_facts = bm25_to_unit(float(abs(row["bm25_facts"] or 0.0)))
            total = (w_nf * score_nf) + (w_ctx * score_ctx) + (w_facts * score_facts)
            scored.append(
                LexicalScore(
                    asset_id=str(row["asset_id"]),
                    total=float(total),
                    namefolder=float(score_nf),
                    context=float(score_ctx),
                    facts=float(score_facts),
                )
            )

        scored.sort(key=lambda x: x.total, reverse=True)
        if filters.min_score > 0:
            scored = [x for x in scored if x.total >= filters.min_score]
        if filters.all_results:
            return scored
        return scored[:limit]

    def _ocr_vector_scores(
        self,
        conn: sqlite3.Connection,
        qvec_text: np.ndarray,
        allowed_ids: set[str],
        model_id: str,
        limit: int,
    ) -> list[tuple[str, float]]:
        if not allowed_ids:
            return []
        placeholders = ",".join("?" for _ in allowed_ids)
        rows = conn.execute(
            f"""
            SELECT v.asset_id, v.embedding
            FROM vectors v
            JOIN assets a ON a.id = v.asset_id
            WHERE v.model_id = ? AND a.is_deleted = 0 AND v.asset_id IN ({placeholders})
            """,
            [model_id, *sorted(allowed_ids)],
        ).fetchall()
        if not rows:
            return []

        q = np.asarray(qvec_text, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn <= 0:
            return []

        out: list[tuple[str, float]] = []
        for row in rows:
            vec = decode_vector(row["embedding"])
            if vec.shape[0] != q.shape[0]:
                continue
            vn = float(np.linalg.norm(vec))
            if vn <= 0:
                continue
            sim = float(np.dot(vec, q) / (vn * qn))
            out.append((str(row["asset_id"]), float((sim + 1.0) / 2.0)))

        out.sort(key=lambda x: x[1], reverse=True)
        return out[:limit]

    def _ocr_lexical_scores(
        self,
        conn: sqlite3.Connection,
        query: str,
        asset_ids: set[str],
    ) -> dict[str, float]:
        q_tokens = [t for t in tokenize(query).split() if len(t) > 1]
        if not q_tokens or not asset_ids:
            return {}
        q_set = set(q_tokens)
        placeholders = ",".join("?" for _ in asset_ids)
        rows = conn.execute(
            f"""
            SELECT asset_id, value_text
            FROM facts
            WHERE fact_type = 'ocr' AND value_text IS NOT NULL AND asset_id IN ({placeholders})
            ORDER BY updated_at DESC, id DESC
            """,
            tuple(sorted(asset_ids)),
        ).fetchall()

        best: dict[str, float] = {}
        for row in rows:
            aid = str(row["asset_id"])
            text = str(row["value_text"] or "")
            tset = set(tokenize(text).split())
            if not tset:
                continue
            overlap = len(q_set.intersection(tset)) / max(1, len(q_set))
            phrase_bonus = 0.15 if query.lower() in text.lower() else 0.0
            score = min(1.0, overlap + phrase_bonus)
            if score > best.get(aid, 0.0):
                best[aid] = score
        return best

    def _vector_scores(
        self,
        conn: sqlite3.Connection,
        query: str,
        filters: SearchFilters,
        limit: int,
    ) -> tuple[list[tuple[str, float]], dict[str, float], dict[str, float], dict[str, float]]:
        embedder = self._embedder()
        index = self._vector_index(embedder.dim)
        qvec_image = embedder.embed_image_query(query)
        qvec_text = embedder.embed_text_query(query)
        allowed = self._allowed_ids(conn, filters)
        image_hits = index.search(qvec_image, k=limit, allowed_asset_ids=allowed)
        image_scores = {h.asset_id: h.score for h in image_hits}

        ocr_model_id = f"{embedder.text_model_id}:ocr"
        ocr_hits = self._ocr_vector_scores(
            conn,
            qvec_text=qvec_text,
            allowed_ids=allowed,
            model_id=ocr_model_id,
            limit=limit,
        )
        ocr_scores = {aid: score for aid, score in ocr_hits}
        ocr_lex_scores = self._ocr_lexical_scores(conn, query=query, asset_ids=allowed)

        image_w, ocr_w = _vector_channel_weights(query)
        merged: dict[str, float] = {}
        for aid, score in image_scores.items():
            merged[aid] = max(merged.get(aid, 0.0), min(1.0, score * image_w))
        for aid, score in ocr_scores.items():
            merged[aid] = max(merged.get(aid, 0.0), min(1.0, score * ocr_w))
        for aid, score in ocr_lex_scores.items():
            merged[aid] = max(merged.get(aid, 0.0), min(1.0, score * 1.15))

        out = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:limit]
        if filters.min_score > 0:
            out = [(aid, s) for aid, s in out if s >= filters.min_score]
        return out, image_scores, ocr_scores, ocr_lex_scores

    def _fetch_candidate_text(self, conn: sqlite3.Connection, asset_id: str) -> CandidateText:
        row = conn.execute(
            """
            SELECT filename_tokens, folder_tokens, context_only_text, facts_only_text
            FROM lexical_docs
            WHERE asset_id = ?
            """,
            (asset_id,),
        ).fetchone()
        if row is None:
            return CandidateText(filename_tokens="", folder_tokens="", contexts="", facts="", metadata="")

        facts = summarize_facts(conn, asset_id)
        exif = facts.get("exif") if isinstance(facts.get("exif"), dict) else {}
        meta_parts = [
            str(exif.get("datetime") or ""),
            str(exif.get("camera_model") or ""),
        ]
        meta_text = " ".join(x for x in meta_parts if x)

        return CandidateText(
            filename_tokens=str(row["filename_tokens"] or ""),
            folder_tokens=str(row["folder_tokens"] or ""),
            contexts=str(row["context_only_text"] or ""),
            facts=str(row["facts_only_text"] or ""),
            metadata=meta_text,
        )

    def _hydrate_results(self, conn: sqlite3.Connection, scored: list[tuple[str, ScoresOutput]]) -> list[dict[str, Any]]:
        if not scored:
            return []

        ids = [asset_id for asset_id, _ in scored]
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"""
            SELECT a.id, a.rel_path, a.abs_path, a.size, a.mtime, a.width, a.height, c.name AS collection
            FROM assets a
            JOIN collections c ON c.id = a.collection_id
            WHERE a.id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        by_id = {str(r["id"]): r for r in rows}

        out: list[dict[str, Any]] = []
        for asset_id, score in scored:
            row = by_id.get(asset_id)
            if row is None:
                continue

            context_rows = contexts_mod.resolve_contexts_for_asset(conn, asset_id)
            contexts = [ContextOutput(virtual_path=c.virtual_path, text=c.text) for c in context_rows]

            facts = summarize_facts(conn, asset_id)
            exif = facts.get("exif") if isinstance(facts.get("exif"), dict) else {}

            facts_model = FactsOutput(
                caption=facts.get("caption"),
                tags=[FactLabelOutput(label=t["label"], conf=t.get("conf")) for t in facts.get("tags", [])],
                objects=[FactLabelOutput(label=t["label"], conf=t.get("conf")) for t in facts.get("objects", [])],
                ocr=facts.get("ocr"),
                exif=exif,
                derived=facts.get("derived") if isinstance(facts.get("derived"), dict) else {},
            )

            metadata = MetadataOutput(
                collection=str(row["collection"]),
                rel_path=str(row["rel_path"]),
                abs_path=str(row["abs_path"]),
                width=row["width"],
                height=row["height"],
                size=int(row["size"]),
                mtime=float(row["mtime"]),
                date=str(exif.get("datetime")) if exif.get("datetime") is not None else None,
                camera=str(exif.get("camera_model")) if exif.get("camera_model") is not None else None,
            )

            score.contributions = _contributions(score)
            result = SearchResultOutput(
                id=asset_id,
                path=str(row["abs_path"]),
                rel_path=str(row["rel_path"]),
                collection=str(row["collection"]),
                metadata=metadata,
                contexts=contexts,
                facts=facts_model,
                scores=score,
            )
            out.append(result.model_dump())
        return out

    def _limit_scored(self, scored: list[tuple[str, ScoresOutput]], filters: SearchFilters) -> list[tuple[str, ScoresOutput]]:
        if filters.min_score > 0:
            scored = [(aid, s) for aid, s in scored if s.overall >= filters.min_score]
        if filters.all_results:
            return scored
        return scored[: filters.num]

    def search(self, query: str, filters: SearchFilters) -> list[dict[str, Any]]:
        limit = 10000 if filters.all_results else max(filters.num, 30)
        with self.db.connect() as conn:
            rows = self._lexical_scores(
                conn,
                query,
                filters,
                limit=limit,
                w_nf=self.config.search.w_namefolder,
                w_ctx=self.config.search.w_context,
                w_facts=self.config.search.w_facts,
            )
            scored = [
                (
                    row.asset_id,
                    ScoresOutput(
                        overall=row.total,
                        lexical_total=row.total,
                        lexical_namefolder=row.namefolder,
                        lexical_context=row.context,
                        lexical_facts=row.facts,
                    ),
                )
                for row in rows
            ]
            scored = self._limit_scored(scored, filters)
            return self._hydrate_results(conn, scored)

    def vsearch(self, query: str, filters: SearchFilters) -> list[dict[str, Any]]:
        limit = 10000 if filters.all_results else max(filters.num * 4, 30)
        with self.db.connect() as conn:
            rows, image_scores, ocr_scores, ocr_lex_scores = self._vector_scores(conn, query, filters, limit=limit)
            scored = [
                (
                    aid,
                    ScoresOutput(
                        overall=score,
                        vector=score,
                        vector_image=image_scores.get(aid),
                        vector_ocr=ocr_scores.get(aid),
                        lexical_ocr=ocr_lex_scores.get(aid),
                    ),
                )
                for aid, score in rows
            ]
            scored = self._limit_scored(scored, filters)
            return self._hydrate_results(conn, scored)

    def query(self, query: str, filters: SearchFilters) -> list[dict[str, Any]]:
        expansions = expand_query_fallback(query, n=self.config.search.query_expansions)
        all_queries = [query, *expansions]

        ranked_lists: list[list[str]] = []
        weights: list[float] = []
        lexical_best: dict[str, LexicalScore] = {}
        vector_best: dict[str, float] = {}
        vector_best_image: dict[str, float] = {}
        vector_best_ocr: dict[str, float] = {}
        ocr_lex_best: dict[str, float] = {}

        with self.db.connect() as conn:
            for idx, q in enumerate(all_queries):
                list_weight = 2.0 if idx == 0 else 1.0
                bias_ctx, bias_facts = _query_channel_bias(q)

                lex_rows = self._lexical_scores(
                    conn,
                    q,
                    filters,
                    limit=self.config.search.top_k_retrieval,
                    w_nf=self.config.search.w_namefolder,
                    w_ctx=self.config.search.w_context * bias_ctx,
                    w_facts=self.config.search.w_facts * bias_facts,
                )
                vec_rows, vec_image_scores, vec_ocr_scores, vec_lex_scores = self._vector_scores(
                    conn, q, filters, limit=self.config.search.top_k_retrieval
                )

                if lex_rows:
                    ranked_lists.append([x.asset_id for x in lex_rows])
                    weights.append(list_weight)
                if vec_rows:
                    ranked_lists.append([x[0] for x in vec_rows])
                    weights.append(list_weight)

                for row in lex_rows:
                    prev = lexical_best.get(row.asset_id)
                    if prev is None or row.total > prev.total:
                        lexical_best[row.asset_id] = row

                for asset_id, score in vec_rows:
                    vector_best[asset_id] = max(vector_best.get(asset_id, 0.0), score)
                    if asset_id in vec_image_scores:
                        vector_best_image[asset_id] = max(
                            vector_best_image.get(asset_id, 0.0), vec_image_scores[asset_id]
                        )
                    if asset_id in vec_ocr_scores:
                        vector_best_ocr[asset_id] = max(vector_best_ocr.get(asset_id, 0.0), vec_ocr_scores[asset_id])
                    if asset_id in vec_lex_scores:
                        ocr_lex_best[asset_id] = max(ocr_lex_best.get(asset_id, 0.0), vec_lex_scores[asset_id])

            if not ranked_lists:
                return []

            fused = fuse_rankings(ranked_lists, k=self.config.search.rrf_k, weights=weights)
            retrieval_ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
            retrieval_norm = _normalize_ranked_rrf(retrieval_ranked)
            candidates = retrieval_ranked[: self.config.search.rerank_candidates]

            reranker = OverlapReranker() if self.config.search.enable_reranker else None

            rescored: list[tuple[str, ScoresOutput]] = []
            for pos, (asset_id, retrieval_score) in enumerate(candidates, start=1):
                candidate = self._fetch_candidate_text(conn, str(asset_id))
                retrieval_score_norm = retrieval_norm.get(str(asset_id), 0.0)
                rr: float | None
                if reranker is None:
                    rr = None
                    final_score = float(retrieval_score_norm)
                else:
                    rr = reranker.score(query, candidate)
                    final_score = blend_retrieval_and_rerank(pos, float(retrieval_score_norm), rr)
                lex = lexical_best.get(str(asset_id))
                score = ScoresOutput(
                    overall=final_score,
                    rrf=float(retrieval_score),
                    lexical_total=lex.total if lex else None,
                    lexical_namefolder=lex.namefolder if lex else None,
                    lexical_context=lex.context if lex else None,
                    lexical_facts=lex.facts if lex else None,
                    lexical_ocr=ocr_lex_best.get(str(asset_id)),
                    vector=vector_best.get(str(asset_id)),
                    vector_image=vector_best_image.get(str(asset_id)),
                    vector_ocr=vector_best_ocr.get(str(asset_id)),
                    rerank=rr,
                )
                rescored.append((str(asset_id), score))

            rescored.sort(key=lambda x: x[1].overall, reverse=True)
            rescored = self._limit_scored(rescored, filters)
            return self._hydrate_results(conn, rescored)

    def _resolve_identifier(self, conn: sqlite3.Connection, identifier: str) -> str | None:
        sid = parse_short_id(identifier)
        if sid is not None:
            aid = short_id_from_int(sid)
            row = conn.execute("SELECT id FROM assets WHERE id = ?", (aid,)).fetchone()
            if row:
                return str(row["id"])

        row = conn.execute(
            "SELECT id FROM assets WHERE abs_path = ? OR rel_path = ?",
            (identifier, identifier),
        ).fetchone()
        if row:
            return str(row["id"])

        if identifier.startswith("qimg://"):
            body = identifier[len("qimg://") :]
            if "/" in body:
                collection, rel = body.split("/", 1)
            else:
                collection, rel = body, ""
            row = conn.execute(
                """
                SELECT a.id
                FROM assets a
                JOIN collections c ON c.id = a.collection_id
                WHERE c.name = ? AND a.rel_path = ?
                """,
                (collection, rel),
            ).fetchone()
            if row:
                return str(row["id"])

        return None

    def get(self, identifier: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            asset_id = self._resolve_identifier(conn, identifier)
            if asset_id is None:
                return None
            rows = self._hydrate_results(conn, [(asset_id, ScoresOutput(overall=1.0))])
            return rows[0] if rows else None

    def multi_get(self, identifiers: list[str] | None = None, glob_pattern: str | None = None) -> list[dict[str, Any]]:
        ids: list[str] = []
        with self.db.connect() as conn:
            if glob_pattern:
                rows = conn.execute("SELECT id, abs_path, rel_path FROM assets WHERE is_deleted = 0").fetchall()
                for row in rows:
                    abs_path = str(row["abs_path"])
                    rel_path = str(row["rel_path"])
                    if fnmatch.fnmatch(abs_path, glob_pattern) or fnmatch.fnmatch(rel_path, glob_pattern):
                        ids.append(str(row["id"]))
            for identifier in identifiers or []:
                found = self._resolve_identifier(conn, identifier)
                if found and found not in ids:
                    ids.append(found)
            scored = [(asset_id, ScoresOutput(overall=1.0)) for asset_id in ids]
            return self._hydrate_results(conn, scored)

    def status(self) -> dict[str, Any]:
        with self.db.connect() as conn:
            counts = {
                "collections": int(conn.execute("SELECT COUNT(*) AS n FROM collections").fetchone()["n"]),
                "assets_active": int(conn.execute("SELECT COUNT(*) AS n FROM assets WHERE is_deleted = 0").fetchone()["n"]),
                "assets_deleted": int(conn.execute("SELECT COUNT(*) AS n FROM assets WHERE is_deleted = 1").fetchone()["n"]),
                "contexts": int(conn.execute("SELECT COUNT(*) AS n FROM contexts").fetchone()["n"]),
                "facts": int(conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]),
                "vectors": int(conn.execute("SELECT COUNT(*) AS n FROM vectors").fetchone()["n"]),
                "lexical_docs": int(conn.execute("SELECT COUNT(*) AS n FROM lexical_docs").fetchone()["n"]),
                "fts_rows": int(conn.execute("SELECT COUNT(*) AS n FROM assets_fts").fetchone()["n"]),
            }
            model = conn.execute(
                """
                SELECT model_id, COUNT(*) AS n
                FROM vectors
                GROUP BY model_id
                ORDER BY n DESC
                LIMIT 1
                """
            ).fetchone()
        counts["vector_model"] = model["model_id"] if model else None
        counts["db_path"] = str(self.db.path)
        counts["vector_dir"] = str(self.config.vector_dir)
        return counts

    def cleanup(self) -> dict[str, Any]:
        embedder = self._embedder()
        index = self._vector_index(embedder.dim)

        with self.db.connect() as conn:
            stats = cleanup_orphans(conn)
            kept = rebuild_index_from_db(conn, index, model_id=embedder.model_id)
            stats["index_vectors"] = kept
        return stats

    def repair(self) -> dict[str, Any]:
        embedder = self._embedder()
        index = self._vector_index(embedder.dim)
        with self.db.connect() as conn:
            count = rebuild_index_from_db(conn, index, model_id=embedder.model_id)
        return {"repaired": count, "model_id": embedder.model_id, "dim": embedder.dim}

    def similar(self, identifier: str, num: int = 10) -> list[dict[str, Any]]:
        embedder = self._embedder()
        index = self._vector_index(embedder.dim)

        with self.db.connect() as conn:
            target_id = self._resolve_identifier(conn, identifier)
            if target_id is None:
                return []
            row = conn.execute(
                "SELECT embedding FROM vectors WHERE asset_id = ? AND model_id = ?",
                (target_id, embedder.model_id),
            ).fetchone()
            if row is None:
                row = conn.execute("SELECT embedding FROM vectors WHERE asset_id = ? LIMIT 1", (target_id,)).fetchone()
            if row is None:
                return []

            qvec = decode_vector(row["embedding"])
            hits = index.search(qvec, k=max(30, num * 3))

            target_phash_row = conn.execute("SELECT phash FROM assets WHERE id = ?", (target_id,)).fetchone()
            target_phash = target_phash_row["phash"] if target_phash_row else None

            scored: dict[str, float] = {}
            for hit in hits:
                if hit.asset_id == target_id:
                    continue
                scored[hit.asset_id] = max(scored.get(hit.asset_id, 0.0), hit.score)

            if target_phash:
                try:
                    from qimg.media.phash import hamming_distance_hex

                    rows = conn.execute(
                        "SELECT id, phash FROM assets WHERE is_deleted = 0 AND phash IS NOT NULL"
                    ).fetchall()
                    for row in rows:
                        aid = str(row["id"])
                        if aid == target_id:
                            continue
                        dist = hamming_distance_hex(target_phash, str(row["phash"]))
                        if dist <= 8:
                            bonus = 1.0 - (dist / 8.0)
                            scored[aid] = max(scored.get(aid, 0.0), bonus)
                except Exception:
                    pass

            ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:num]
            wrapped = [(aid, ScoresOutput(overall=score, vector=score)) for aid, score in ranked]
            return self._hydrate_results(conn, wrapped)

    def get_collection(self, name: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            coll = get_collection_by_name(conn, name)
        if coll is None:
            return None
        return {
            "id": coll.id,
            "name": coll.name,
            "root_path": coll.root_path,
            "mask": coll.mask,
            "created_at": coll.created_at,
        }
