import json
from pathlib import Path

from genpy.training.config import load_training_config


def test_review_classifies_legacy_test_and_clean_challenge() -> None:
    audit = json.loads(Path("reports/checkpoint_8_instruction_data_audit.json").read_text(encoding="utf-8"))
    challenge = json.loads(Path("reports/checkpoint_8_challenge_contamination.json").read_text(encoding="utf-8"))
    assert audit["evaluation_classification"]["original_production_test"] == "LEGACY_IN_DISTRIBUTION_CONTAMINATION_RISK"
    assert challenge["status"] == "CLEAN_GENERALIZATION"
    assert audit["cross_split"]["train_test_solution_overlap"] == 699
    assert audit["cross_split"]["train_test_template_overlap"] == 51


def test_budget_recommends_shortest_zero_truncation_context() -> None:
    budget = json.loads(Path("reports/checkpoint_8_sft_budget.json").read_text(encoding="utf-8"))
    config = load_training_config("configs/sft_200m_kaggle.yaml")
    assert budget["recommended_sequence_length"] == 256
    assert budget["formatted_sequence_length"]["p95"] == 99.0
    assert budget["formatted_sequence_length"]["p99"] == 103.0
    assert budget["formatted_sequence_length"]["max"] == 158.0
    assert config.training.sequence_length == 256
    assert config.training.max_steps == 11250
    assert budget["sequence_analysis"]["256"]["truncation_count_estimate"] == 0
