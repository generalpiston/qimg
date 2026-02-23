from __future__ import annotations


def bm25_to_unit(score: float) -> float:
    score = abs(score)
    return score / (score + 10.0)


def distance_to_similarity(distance: float) -> float:
    # Works well for cosine-distance style values.
    return 1.0 / (1.0 + max(0.0, distance))


def cosine_to_unit(cosine_sim: float) -> float:
    return max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))
