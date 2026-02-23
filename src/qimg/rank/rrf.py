from __future__ import annotations

from collections import defaultdict
from typing import Hashable


def fuse_rankings(
    ranked_lists: list[list[Hashable]],
    k: int = 60,
    weights: list[float] | None = None,
) -> dict[Hashable, float]:
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    fused: dict[Hashable, float] = defaultdict(float)

    for list_idx, ranked in enumerate(ranked_lists):
        w = weights[list_idx] if list_idx < len(weights) else 1.0
        for pos, item_id in enumerate(ranked, start=1):
            fused[item_id] += w * (1.0 / (k + pos))
            if pos == 1:
                fused[item_id] += 0.05
            elif pos in (2, 3):
                fused[item_id] += 0.02

    return dict(fused)
