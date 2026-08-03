from __future__ import annotations

from genpy.data.near_dedup import deduplicate_near, jaccard_similarity, token_shingles
from tests.unit._helpers import make_record


def test_near_duplicate_clustering(tmp_path) -> None:  # type: ignore[no-untyped-def]
    left = "\n".join(f"value_{index} = {index}" for index in range(60)) + "\n"
    right = left.replace("value_59 = 59", "value_59 = 60")
    assert jaccard_similarity(token_shingles(left, 5), token_shingles(right, 5)) >= 0.85
    winners = deduplicate_near(
        [make_record(left, identity="a.py"), make_record(right, identity="b.py")],
        tmp_path,
    )
    assert len(winners) == 1
