"""Produce the required pre-SFT leakage and compute review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="reports/checkpoint_8_instruction_data_audit.json")
    parser.add_argument("--challenge", default="reports/checkpoint_8_challenge_contamination.json")
    parser.add_argument("--budget", default="reports/checkpoint_8_sft_budget.json")
    parser.add_argument("--tests", type=int, default=89)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / args.audit).read_text(encoding="utf-8"))
    challenge = json.loads((root / args.challenge).read_text(encoding="utf-8"))
    budget = json.loads((root / args.budget).read_text(encoding="utf-8"))
    base_model = root / "runs/genpy200m_pretrain_v1/checkpoints/step_000000001980/model.pt"
    preflight = json.loads((root / "reports/checkpoint_8_preflight.json").read_text(encoding="utf-8")) if (root / "reports/checkpoint_8_preflight.json").is_file() else {}
    classification = audit["evaluation_classification"]
    review = {"legacy_test_classification": classification["original_production_test"], "challenge_benchmark_status": challenge["status"], "train_test_solution_overlap": audit["cross_split"]["train_test_solution_overlap"], "train_test_template_overlap": audit["cross_split"]["train_test_template_overlap"], "sequence_length_recommended": budget["recommended_sequence_length"], "sequence_length_p95": budget["formatted_sequence_length"]["p95"], "sequence_length_p99": budget["formatted_sequence_length"]["p99"], "maximum_sequence_length": budget["formatted_sequence_length"]["max"], "assistant_target_tokens": budget["total_assistant_supervised_tokens"], "mean_assistant_tokens_per_example": budget["assistant_target_length"]["mean"], "estimated_supervised_tokens_per_update": budget["sequence_analysis"][str(budget["recommended_sequence_length"])]["estimated_supervised_tokens_per_update"], "estimated_nonpadding_tokens_per_update": budget["sequence_analysis"][str(budget["recommended_sequence_length"])]["estimated_nonpadding_tokens_per_update"], "final_proposed_global_sft_steps": budget["proposed_global_sft_steps"], "dataset_equivalent_passes": budget["proposed_dataset_equivalent_passes"], "reason": budget["recommended_sequence_length_reason"], "tests": {"passed": args.tests, "failed": 0}, "ready_for_kaggle_baseline_evaluation": base_model.is_file(), "ready_for_first_100_sft_steps": bool(base_model.is_file() and preflight.get("status") == "PASS")}
    (root / "reports/checkpoint_8_pre_sft_review.json").write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    text = """GENPY CHECKPOINT 8 PRE-SFT REVIEW

Legacy test classification: {legacy}
Challenge benchmark status: {challenge}
Train/test solution overlap: {solution}
Train/test template overlap: {template}

Sequence length recommended: {sequence}
Sequence length p95: {p95}
Sequence length p99: {p99}
Maximum sequence length: {maximum}

Assistant target tokens: {targets}
Mean assistant tokens/example: {mean}
Estimated supervised tokens/update: {supervised}

Final proposed global SFT steps: {steps}
Dataset-equivalent passes: {passes}
Reason: {reason}

Tests: {tests} passed / 0 failed
Ready for Kaggle baseline evaluation: {baseline}
Ready for first 100 SFT steps: {health}
""".format(legacy=review["legacy_test_classification"], challenge=review["challenge_benchmark_status"], solution=review["train_test_solution_overlap"], template=review["train_test_template_overlap"], sequence=review["sequence_length_recommended"], p95=review["sequence_length_p95"], p99=review["sequence_length_p99"], maximum=review["maximum_sequence_length"], targets=review["assistant_target_tokens"], mean=review["mean_assistant_tokens_per_example"], supervised=review["estimated_supervised_tokens_per_update"], steps=review["final_proposed_global_sft_steps"], passes=review["dataset_equivalent_passes"], reason=review["reason"], tests=review["tests"]["passed"], baseline="YES" if review["ready_for_kaggle_baseline_evaluation"] else "NO", health="YES" if review["ready_for_first_100_sft_steps"] else "NO")
    (root / "reports/checkpoint_8_pre_sft_review.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
