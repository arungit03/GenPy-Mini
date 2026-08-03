from __future__ import annotations

from genpy.data.exact_dedup import deduplicate_exact
from tests.unit._helpers import make_record


def test_exact_duplicate_keeps_one_record(tmp_path) -> None:  # type: ignore[no-untyped-def]
    text = "def add(left, right):\n    return left + right\n"
    records = [make_record(text, identity="a.py"), make_record(text, identity="b.py")]
    winners = deduplicate_exact(records, tmp_path)
    assert len(winners) == 1
    assert (tmp_path / "exact.jsonl").read_text(encoding="utf-8").count("\n") == 1
