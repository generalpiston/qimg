from qimg.rank.rrf import fuse_rankings


def test_rrf_fuse_with_weights_and_bonus() -> None:
    ranked_lists = [
        [1, 2, 3],
        [2, 3, 4],
    ]
    scores = fuse_rankings(ranked_lists, k=60, weights=[2.0, 1.0])

    assert 1 in scores
    assert 2 in scores
    assert scores[2] > scores[1]  # id=2 gets strong support in both lists.
    assert scores[2] > scores[3]
