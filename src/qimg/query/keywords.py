from __future__ import annotations

import re

STOPWORDS = {
    "a",
    "an",
    "the",
    "on",
    "in",
    "at",
    "with",
    "of",
    "for",
    "to",
    "and",
    "or",
    "from",
    "by",
    "is",
    "are",
    "this",
    "that",
}


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text)]
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok in STOPWORDS or len(tok) <= 1:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return out
