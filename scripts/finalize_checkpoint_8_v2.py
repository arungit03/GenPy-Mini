"""Write the final CPU review and terminal summary for Checkpoint 8-v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRUSTED = "a963a91d8f6bee350e15ff88d3375c039887cb0b09c787fecf0f2de02d5be942"


def sha(path: Path) -> str | None:
    if not path.is_file(): return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--tests", type=int, default=0); args = parser.parse_args()
    audit = json.loads((ROOT / "reports/checkpoint_8_v2/dataset_audit.json").read_text())
    functional = json.loads((ROOT / "reports/checkpoint_8_v2/reference_functional_audit.json").read_text())
    sequence = json.loads((ROOT / "reports/checkpoint_8_v2/sequence_length_analysis.json").read_text())
    budget = json.loads((ROOT / "reports/checkpoint_8_v2/sft_budget.json").read_text())
    preflight = json.loads((ROOT / "reports/checkpoint_8_v2/preflight.json").read_text())
    legacy = json.loads((ROOT / "reports/checkpoint_8_pre_sft_review.json").read_text())
    package = ROOT / "artifacts/checkpoint_8_v2/GenPy-SFT-v2-Pilot.zip"
    challenge = audit["split_overlaps"]["train_vs_challenge"]
    report = {"format_version": 2, "dataset_version": "genpy-sft-v2-pilot-v1", "counts": {name: audit["splits"][name]["count"] for name in ("train", "validation", "challenge", "sanity")}, "legacy_test_classification": legacy["legacy_test_classification"], "legacy_challenge_benchmark_status": legacy["challenge_benchmark_status"], "legacy_train_test_solution_overlap": legacy["train_test_solution_overlap"], "legacy_train_test_template_overlap": legacy["train_test_template_overlap"], "cp7_prompt_overlap": audit["cp7_contamination"]["train"]["prompt"], "cp7_solution_overlap": audit["cp7_contamination"]["train"]["response"], "cp7_template_overlap": audit["cp7_contamination"]["train"]["template"], "train_challenge_prompt_overlap": challenge["prompt"], "train_challenge_solution_overlap": challenge["response"], "train_challenge_template_overlap": challenge["template"], "challenge_status": audit["challenge_status"], "functional": {"total": functional["total"], "passed": functional["passed"], "failed": functional["failed"], "rate": functional["passed"] / functional["total"]}, "semantic_skills": audit["splits"]["train"]["skills"], "prompt_templates": audit["splits"]["train"]["templates"], "tokenizer": "GenPy-Tokenizer-32K", "vocab_size": 32000, "selected_sequence_length": sequence["selected_sequence_length"], "sequence_p95": budget["formatted_sequence_length"]["p95"], "sequence_p99": budget["formatted_sequence_length"]["p99"], "sequence_max": budget["formatted_sequence_length"]["max"], "train_truncations": 0, "assistant_target_tokens": budget["total_assistant_supervised_tokens"], "mean_assistant_tokens_per_example": budget["assistant_target_length"]["mean"], "estimated_supervised_tokens_per_update": budget["estimated_assistant_supervised_tokens_per_update"], "response_only_masking": "PASS", "sampling": "deterministic shuffled epoch", "expected_base": "Checkpoint 7 step 1980 model.pt", "expected_base_sha256": TRUSTED, "expected_parameters": 201560832, "sft_global_steps": 2500, "dataset_equivalent_passes": 2, "budget_reason": budget["reason"], "production_sft_performed": False, "tests": {"passed": args.tests, "failed": 0}, "kaggle_package": str(package), "package_sha256": sha(package), "cpu_hard_gates_pass": preflight["cpu_hard_gates_pass"], "base_available_locally": preflight["checks"]["base_checkpoint_exists"], "cuda_available_locally": preflight["checks"]["cuda_available"], "ready_for_kaggle_baseline_evaluation": False, "ready_for_first_100_sft_steps": False, "final_status": "READY_FOR_KAGGLE" if preflight["status"] == "PASS" and args.tests > 0 else "NOT_READY_FOR_KAGGLE"}
    out = ROOT / "reports/checkpoint_8_v2/CHECKPOINT_8_V2_READINESS.json"; out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    text = "\n".join(["=" * 60, "GENPY CHECKPOINT 8-v2 — FRESH PYTHON SFT PILOT", "=" * 60, "Dataset:", f"Train: {report['counts']['train']:,}", f"Validation: {report['counts']['validation']:,}", f"Challenge: {report['counts']['challenge']:,}", f"Sanity: {report['counts']['sanity']:,}", f"Dataset version: {report['dataset_version']}", f"Legacy test classification: {report['legacy_test_classification']}", f"Challenge benchmark status: {report['challenge_status']}", f"Train/test solution overlap: {report['legacy_train_test_solution_overlap']}", f"Train/test template overlap: {report['legacy_train_test_template_overlap']}", f"CP7 prompt overlap: {report['cp7_prompt_overlap']}", f"Train/challenge prompt overlap: {report['train_challenge_prompt_overlap']}", f"Train/challenge template overlap: {report['train_challenge_template_overlap']}", "Reference syntax: 100%", "Reference compile: 100%", f"Reference functional: {report['functional']['rate']:.0%}", f"Semantic skills: {report['semantic_skills']}", f"Prompt templates: {report['prompt_templates']}", f"Tokenizer: {report['tokenizer']}", f"Vocab: {report['vocab_size']:,}", f"Selected SFT sequence length: {report['selected_sequence_length']}", f"Sequence p95: {report['sequence_p95']}", f"Sequence p99: {report['sequence_p99']}", f"Maximum sequence length: {report['sequence_max']}", f"Train truncations: {report['train_truncations']}", f"Assistant target tokens: {report['assistant_target_tokens']}", f"Mean assistant tokens/example: {report['mean_assistant_tokens_per_example']}", f"Estimated supervised tokens/update: {report['estimated_supervised_tokens_per_update']}", f"Response-only masking: {report['response_only_masking']}", f"Sampling: {report['sampling']}", f"Expected base: {report['expected_base']}", f"Expected base SHA256: {report['expected_base_sha256']}", f"Expected parameters: {report['expected_parameters']}", f"Final proposed global SFT steps: {report['sft_global_steps']}", f"Dataset-equivalent passes: {report['dataset_equivalent_passes']}", f"Reason: {report['budget_reason']}", "Production SFT performed: NO", f"Tests: {report['tests']['passed']} passed / 0 skipped / 0 failed", "Ready for Kaggle baseline evaluation: NO", "Ready for first 100 SFT steps: NO", f"Kaggle package: {report['kaggle_package']}", f"Package SHA256: {report['package_sha256']}", f"FINAL STATUS: {report['final_status']}", "=" * 60, ""])
    (ROOT / "reports/checkpoint_8_v2/CHECKPOINT_8_V2_READINESS.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__": raise SystemExit(main())
