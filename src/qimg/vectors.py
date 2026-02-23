from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np

from qimg.db import decode_vector, encode_vector
from qimg.ids import asset_id_to_label, label_to_asset_id
from qimg.rank.normalize import cosine_to_unit
from qimg.util.time import now_iso


@dataclass(slots=True)
class VectorHit:
    asset_id: str
    score: float


class VectorIndex:
    def __init__(
        self,
        dim: int,
        index_path: Path,
        fallback_path: Path,
    ):
        self.dim = dim
        self.index_path = index_path
        self.fallback_path = fallback_path
        self.ids: np.ndarray = np.array([], dtype=np.int64)
        self.vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._hnsw = None
        self._hnsw_available = False
        self._load_optional_hnsw()
        self.load()

    def _load_optional_hnsw(self) -> None:
        try:
            import hnswlib  # type: ignore

            self._hnsw = hnswlib
            self._hnsw_available = True
        except Exception:
            self._hnsw_available = False

    def load(self) -> None:
        if self.fallback_path.exists():
            data = np.load(self.fallback_path)
            self.ids = data["ids"].astype(np.int64)
            self.vectors = data["vectors"].astype(np.float32)
        else:
            self.ids = np.array([], dtype=np.int64)
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)

        if self.vectors.shape[0] > 0 and self.vectors.shape[1] != self.dim:
            self.ids = np.array([], dtype=np.int64)
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)

        if self._hnsw_available:
            self._rebuild_hnsw()

    def _rebuild_hnsw(self) -> None:
        assert self._hnsw is not None
        hnsw = self._hnsw.Index(space="cosine", dim=self.dim)
        n = int(max(1, self.vectors.shape[0]))
        hnsw.init_index(max_elements=n + 32, ef_construction=200, M=16)
        if self.vectors.shape[0] > 0:
            hnsw.add_items(self.vectors, self.ids)
        hnsw.set_ef(100)
        self._index = hnsw
        if self.index_path.parent:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(self.index_path))

    def _persist_npz(self) -> None:
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.fallback_path, ids=self.ids, vectors=self.vectors)

    def upsert(self, asset_id: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape[0] != self.dim:
            raise ValueError(f"vector dim mismatch: expected {self.dim}, got {vector.shape[0]}")

        label = asset_id_to_label(asset_id)
        idx = np.where(self.ids == label)[0]
        if idx.size > 0:
            self.vectors[idx[0]] = vector
        else:
            self.ids = np.append(self.ids, np.int64(label))
            if self.vectors.shape[0] == 0:
                self.vectors = vector.reshape(1, -1)
            else:
                self.vectors = np.vstack([self.vectors, vector])

    def remove_asset_ids(self, asset_ids: Iterable[str]) -> None:
        labels = {asset_id_to_label(i) for i in asset_ids}
        if not labels or self.ids.size == 0:
            return
        keep_mask = np.array([int(i) not in labels for i in self.ids], dtype=bool)
        self.ids = self.ids[keep_mask]
        self.vectors = self.vectors[keep_mask]

    def persist(self) -> None:
        self._persist_npz()
        if self._hnsw_available:
            self._rebuild_hnsw()

    def search(self, query_vec: np.ndarray, k: int = 10, allowed_asset_ids: set[str] | None = None) -> list[VectorHit]:
        if self.vectors.shape[0] == 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32)
        if q.shape[0] != self.dim:
            raise ValueError(f"query vector dim mismatch: expected {self.dim}, got {q.shape[0]}")

        if allowed_asset_ids:
            allowed_labels = {asset_id_to_label(aid) for aid in allowed_asset_ids}
            mask = np.array([int(i) in allowed_labels for i in self.ids], dtype=bool)
            ids = self.ids[mask]
            vecs = self.vectors[mask]
            if ids.size == 0:
                return []
            sims = vecs @ q
            order = np.argsort(-sims)[:k]
            return [
                VectorHit(asset_id=label_to_asset_id(int(ids[i])), score=float(cosine_to_unit(float(sims[i]))))
                for i in order
            ]

        if self._hnsw_available and hasattr(self, "_index"):
            labels, distances = self._index.knn_query(q.reshape(1, -1), k=min(k, self.ids.size))
            out: list[VectorHit] = []
            for lid, dist in zip(labels[0], distances[0], strict=False):
                sim = 1.0 - float(dist)
                out.append(VectorHit(asset_id=label_to_asset_id(int(lid)), score=float(cosine_to_unit(sim))))
            return out

        sims = self.vectors @ q
        order = np.argsort(-sims)[:k]
        return [
            VectorHit(asset_id=label_to_asset_id(int(self.ids[i])), score=float(cosine_to_unit(float(sims[i]))))
            for i in order
        ]


def fetch_vectors_from_db(conn: sqlite3.Connection, model_id: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    if model_id:
        rows = conn.execute(
            """
            SELECT v.asset_id, v.embedding
            FROM vectors v
            JOIN assets a ON a.id = v.asset_id
            WHERE v.model_id = ? AND a.is_deleted = 0
            ORDER BY v.asset_id
            """,
            (model_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT v.asset_id, v.embedding
            FROM vectors v
            JOIN assets a ON a.id = v.asset_id
            WHERE a.is_deleted = 0
            ORDER BY v.asset_id
            """
        ).fetchall()

    labels: list[int] = []
    vecs: list[np.ndarray] = []
    for row in rows:
        labels.append(asset_id_to_label(str(row["asset_id"])))
        vecs.append(decode_vector(row["embedding"]))

    if not vecs:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    dim = vecs[0].shape[0]
    return np.array(labels, dtype=np.int64), np.vstack([v.reshape(1, dim) for v in vecs]).astype(np.float32)


def upsert_vector_row(conn: sqlite3.Connection, asset_id: str, model_id: str, vector: np.ndarray) -> None:
    blob, _ = encode_vector(vector)
    conn.execute(
        """
        INSERT INTO vectors(asset_id, model_id, embedding, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(asset_id, model_id) DO UPDATE SET
          embedding=excluded.embedding,
          updated_at=excluded.updated_at
        """,
        (asset_id, model_id, blob, now_iso()),
    )


def rebuild_index_from_db(conn: sqlite3.Connection, index: VectorIndex, model_id: str | None = None) -> int:
    labels, vectors = fetch_vectors_from_db(conn, model_id=model_id)
    if labels.size == 0:
        index.ids = np.array([], dtype=np.int64)
        index.vectors = np.zeros((0, index.dim), dtype=np.float32)
        index.persist()
        return 0

    if vectors.shape[1] != index.dim:
        raise ValueError(
            f"cannot rebuild index: vector dim={vectors.shape[1]} but index dim={index.dim}; check configured embed model"
        )

    index.ids = labels
    index.vectors = vectors
    index.persist()
    return int(labels.size)
