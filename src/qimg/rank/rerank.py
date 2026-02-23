from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(slots=True)
class CandidateText:
    filename_tokens: str
    folder_tokens: str
    contexts: str
    facts: str
    metadata: str


class BaseReranker:
    def score(self, query: str, candidate: CandidateText) -> float:
        raise NotImplementedError


class DisabledReranker(BaseReranker):
    def score(self, query: str, candidate: CandidateText) -> float:
        return 0.5


class OverlapReranker(BaseReranker):
    """Lightweight reranker for local-only deployments."""

    def score(self, query: str, candidate: CandidateText) -> float:
        q_tokens = {t.lower() for t in re.findall(r"[a-zA-Z0-9]+", query) if len(t) > 1}
        if not q_tokens:
            return 0.0

        def overlap_score(text: str) -> float:
            toks = {t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text) if len(t) > 1}
            if not toks:
                return 0.0
            return len(q_tokens.intersection(toks)) / max(1, len(q_tokens))

        nf = overlap_score(f"{candidate.filename_tokens} {candidate.folder_tokens}")
        ctx = overlap_score(candidate.contexts)
        facts = overlap_score(candidate.facts)
        meta = overlap_score(candidate.metadata)

        # Context and facts are distinct channels with independent contribution.
        score = (0.25 * nf) + (0.35 * ctx) + (0.30 * facts) + (0.10 * meta)
        return min(1.0, score)


def blend_retrieval_and_rerank(position: int, retrieval_score: float, rerank_score: float) -> float:
    if position <= 3:
        return 0.75 * retrieval_score + 0.25 * rerank_score
    if position <= 10:
        return 0.60 * retrieval_score + 0.40 * rerank_score
    return 0.40 * retrieval_score + 0.60 * rerank_score
