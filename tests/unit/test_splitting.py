from __future__ import annotations

from dataclasses import replace

from genpy.data.splitting import choose_split, leakage_report, split_pretraining_records
from tests.unit._helpers import make_record


def test_deterministic_group_aware_splitting() -> None:
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    assert choose_split("family", ratios, 1337) == choose_split("family", ratios, 1337)
    records = [
        make_record("value = 1\n", identity="a.py", group="same/repository"),
        make_record("value = 2\n", identity="b.py", group="same/repository"),
    ]
    assigned = list(split_pretraining_records(records, ratios, 1337))
    assert assigned[0].split == assigned[1].split


def test_cross_split_exact_leakage_fails_validation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    record = make_record("answer = 42\n", identity="a.py", group="project/a")
    leaked = replace(record, record_id="leaked-copy", split_group="project/b", split="test")
    report = leakage_report([record, leaked], tmp_path / "leakage.sqlite3")
    assert report["exact_cross_split_duplicates"] == 1
    assert report["passed"] is False
