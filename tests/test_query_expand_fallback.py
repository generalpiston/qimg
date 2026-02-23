from qimg.query.expand import expand_query_fallback


def test_query_expand_fallback_is_deterministic() -> None:
    query = "minimalist white kitchen"
    a = expand_query_fallback(query, n=2)
    b = expand_query_fallback(query, n=2)
    assert a == b
    assert len(a) == 2
    assert all("kitchen" in x for x in a)
