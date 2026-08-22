import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_sft_v2_dataset import validate_record
from scripts.generate_sft_v2_pilot import build_split, cases, code_for
from scripts.preflight_sft_v2 import EXPECTED_BASE_SHA256, EXPECTED_PARAMETERS


def test_v2_pilot_counts_and_required_fields():
    rows = build_split("train", 100, 42)
    assert len(rows) == 100
    required = {"id", "instruction", "input", "response", "response_type", "category", "skill_id", "prompt_template_id", "difficulty", "task_style", "function_name", "test_cases", "provenance"}
    assert all(required <= row.keys() for row in rows)
    assert all(len(row["test_cases"]) >= 3 for row in rows)
    assert all(not validate_record(row) for row in rows)


def test_v2_generation_is_deterministic_and_challenge_templates_are_disjoint():
    first = build_split("train", 100, 42)
    second = build_split("train", 100, 42)
    other = build_split("train", 100, 43)
    assert first == second
    assert first != other
    challenge = build_split("challenge", 100, 42)
    assert not ({row["prompt_template_id"] for row in first} & {row["prompt_template_id"] for row in challenge})


def test_v2_function_catalog_covers_challenge_cases():
    kinds = ["roman", "version", "decode_runs", "base_digits", "border", "nearest", "snake", "unique_window", "reverse", "balanced", "merge_sorted", "prefix_products", "running", "histogram", "median"]
    for kind in kinds:
        assert len(cases(kind)) >= 3
        compile(code_for(kind, "solve"), "<test>", "exec")


def test_v2_reports_and_final_budget_are_frozen():
    sequence = json.loads((ROOT / "reports/checkpoint_8_v2/sequence_length_analysis.json").read_text())
    budget = json.loads((ROOT / "reports/checkpoint_8_v2/sft_budget.json").read_text())
    assert sequence["selected_sequence_length"] == 256
    assert sequence["candidates"]["256"]["truncation_count"] == 0
    assert budget["proposed_global_sft_steps"] == 2500
    assert budget["total_assistant_supervised_tokens"] == 465144


def test_v2_trusted_base_is_model_pt_hash_and_parameter_count():
    assert EXPECTED_BASE_SHA256 == "a963a91d8f6bee350e15ff88d3375c039887cb0b09c787fecf0f2de02d5be942"
    assert EXPECTED_PARAMETERS == 201560832
