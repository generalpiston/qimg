from __future__ import annotations

from qimg.query.keywords import extract_keywords


def expand_query_fallback(query: str, n: int = 2) -> list[str]:
    keywords = extract_keywords(query, limit=10)
    kw = " ".join(keywords)

    candidates = [
        f"{query} objects subjects details {kw}".strip(),
        f"{query} style composition setting time period {kw}".strip(),
        f"{query} location scene lighting mood {kw}".strip(),
    ]

    out: list[str] = []
    for c in candidates:
        if c not in out and c != query:
            out.append(c)
        if len(out) >= n:
            break
    return out
