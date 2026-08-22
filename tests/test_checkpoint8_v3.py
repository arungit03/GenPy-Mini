import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_sft_v3_dataset import REQUIRED, validate
from scripts.generate_sft_v3_semantic import build_normal, sanity_rows


def rows(name):
    return [json.loads(line) for line in (ROOT / "data/instruction/python_v3" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line]


def test_v3_exact_counts_and_stable_solve_interface():
    assert {name: len(rows(name)) for name in ("train", "validation", "challenge", "sanity")} == {"train": 3000, "validation": 300, "challenge": 200, "sanity": 20}
    assert all(row["function_name"] == "solve" for name in ("train", "validation", "challenge", "sanity") for row in rows(name))


def test_v3_schema_and_five_test_minimum():
    all_rows = [row for name in ("train", "validation", "challenge", "sanity") for row in rows(name)]
    assert all(REQUIRED <= row.keys() and len(row["test_cases"]) >= 5 and not validate(row) for row in all_rows)


def test_v3_no_artificial_user_visible_identifiers():
    pattern = re.compile(r"(?:_train_|_validation_|_challenge_|Scenario\s+\d+|\b(?:task|example\s+id)\s+\d+)", re.I)
    assert not any(pattern.search(row["instruction"] + row["input"] + row["response"]) for name in ("train", "validation", "challenge", "sanity") for row in rows(name))


def test_v3_prompt_and_template_split_isolation():
    data = {name: rows(name) for name in ("train", "validation", "challenge", "sanity")}
    prompts = {name: {row["instruction"].lower() for row in values} for name, values in data.items()}
    templates = {name: {row["prompt_template_id"] for row in values} for name, values in data.items()}
    assert not (prompts["train"] & prompts["validation"] or prompts["train"] & prompts["challenge"] or prompts["validation"] & prompts["challenge"])
    assert not (templates["train"] & templates["validation"] or templates["train"] & templates["challenge"] or templates["validation"] & templates["challenge"])


def test_v3_challenge_and_sanity_exclusion_metadata():
    assert all(row["provenance"]["training_excluded"] for row in rows("challenge"))
    assert all(row["provenance"]["optimizer_excluded"] and row["provenance"]["validation_excluded"] for row in rows("sanity"))


def test_v3_deterministic_generator_and_manifest_hashes():
    first = build_normal("train", 80, 42)
    second = build_normal("train", 80, 42)
    other = build_normal("train", 80, 43)
    assert first == second
    assert first != other
    manifest = json.loads((ROOT / "data/instruction/python_v3/DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    for name in ("train", "validation", "challenge", "sanity"):
        assert manifest["files"][name]["count"] == len(rows(name))
        assert len(manifest["files"][name]["sha256"]) == 64


def test_v3_functional_report_is_complete():
    report = json.loads((ROOT / "reports/checkpoint_8_v3/reference_functional_audit.json").read_text(encoding="utf-8"))
    assert report["all_pass"] is True
    assert report["records_tested"] == 3520
    assert report["individual_test_cases_executed"] == 17600
    assert report["individual_test_cases_failed"] == 0
