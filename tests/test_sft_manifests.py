import json
from pathlib import Path

from audit_instruction_dataset import audit_split
from genpy.data.io import sha256_file


def test_sft_cache_manifest_hashes_and_test_is_immutable() -> None:
    manifest = json.loads(Path("data/instruction/tokenized/SFT_TOKEN_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["test_split_immutable"] is True
    for name in ("train", "validation", "test"):
        entry = manifest["splits"][name]
        root = Path("data/instruction/tokenized")
        assert sha256_file(root / f"{name}.input_ids.bin") == entry["input_ids_sha256"]
        assert sha256_file(root / f"{name}.labels.bin") == entry["labels_sha256"]
        assert entry["truncation_count"] == 0


def test_audit_reports_duplicate_categories_without_discarding(tmp_path) -> None:
    source = tmp_path / "split.jsonl"
    rows = [
        {"instruction": "Solve 1", "input": "", "response": "def solve():\n    return 1", "family_id": "a"},
        {"instruction": "Solve 2", "input": "", "response": "def solve():\n    return 1", "family_id": "b"},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    stats, records = audit_split(source)
    assert stats["records_seen"] == 2
    assert stats["solution_duplicate_records"] == 1
    assert len(records) == 2
