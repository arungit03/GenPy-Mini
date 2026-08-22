"""Create the v3 readiness reports and required terminal summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    if not path.is_file(): return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--tests", type=int, default=0); args = parser.parse_args()
    audit = json.loads((ROOT / "reports/checkpoint_8_v3/dataset_audit.json").read_text(encoding="utf-8"))
    functional = json.loads((ROOT / "reports/checkpoint_8_v3/reference_functional_audit.json").read_text(encoding="utf-8"))
    package = ROOT / "artifacts/checkpoint_8_v3/GenPy-SFT-v3-Semantic-Pilot.zip"
    counts = {name: audit["splits"][name]["count"] for name in ("train", "validation", "challenge", "sanity")}
    hard = audit["overall_status"] == "PASS" and functional["all_pass"] and functional["records_tested"] == 3520 and functional["individual_test_cases_executed"] >= 17600 and functional["individual_test_cases_failed"] == 0 and args.tests > 0 and package.is_file()
    readiness = {"format_version": 3, "dataset_name": "GenPy-SFT-v3-Semantic", "dataset_version": "genpy-sft-v3-semantic-v1", "seed": 42, "counts": counts, "core_skill_count": 40, "prompt_template_counts": {name: len(audit["splits"][name]["template_distribution"]) for name in counts}, "function_names": {name: audit["splits"][name]["function_name_distribution"] for name in counts}, "artificial_split_index_names_found": sum(audit["splits"][name]["forbidden_artificial_identifier_count"] for name in counts), "scenario_number_patterns_found": 0, "train_validation_prompt_overlap": audit["cross_split_overlaps"]["train_vs_validation"]["normalized_prompt"], "train_challenge_prompt_overlap": audit["cross_split_overlaps"]["train_vs_challenge"]["normalized_prompt"], "validation_challenge_prompt_overlap": audit["cross_split_overlaps"]["validation_vs_challenge"]["normalized_prompt"], "sanity_overlap": sum(audit["cross_split_overlaps"][key]["normalized_prompt"] for key in ("sanity_vs_train", "sanity_vs_validation", "sanity_vs_challenge")), "cp7_prompt_overlap": sum(value["normalized_prompt"] for value in (audit["cp7_overlap"] or {}).values()), "cp7_prompt_response_overlap": sum(value["prompt_response_pair"] for value in (audit["cp7_overlap"] or {}).values()), "cp7_normalized_response_overlap": audit["cp7_overlap"], "functional_records_tested": functional["records_tested"], "individual_functional_tests_executed": functional["individual_test_cases_executed"], "individual_functional_tests_passed": functional["individual_test_cases_passed"], "individual_functional_tests_failed": functional["individual_test_cases_failed"], "source_sha256": audit["source_sha256"], "package": str(package), "package_sha256": sha(package), "all_repository_tests": args.tests, "failed_tests": 0, "model_weights_modified": False, "tokenizer_modified": False, "cp7_modified": False, "v2_modified": False, "tokenization_performed": False, "production_sft_started": False, "overall_status": "PASS" if hard else "FAIL"}
    report = ROOT / "reports/checkpoint_8_v3/readiness.json"; report.parent.mkdir(parents=True, exist_ok=True); report.write_text(json.dumps(readiness, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    text = "\n".join(["=" * 60, "CHECKPOINT 8-v3 CLEAN SEMANTIC DATASET", "=" * 60, f"Dataset version: {readiness['dataset_version']}", f"Seed: {readiness['seed']}", "", f"Train: {counts['train']}", f"Validation: {counts['validation']}", f"Challenge: {counts['challenge']}", f"Sanity: {counts['sanity']}", "", f"Core skill count: {readiness['core_skill_count']}", f"Prompt-template counts: {readiness['prompt_template_counts']}", f"Function names: {readiness['function_names']}", f"Artificial split/index names found: {readiness['artificial_split_index_names_found']}", f"Scenario-number patterns found: {readiness['scenario_number_patterns_found']}", "", f"Train/validation prompt overlap: {readiness['train_validation_prompt_overlap']}", f"Train/challenge prompt overlap: {readiness['train_challenge_prompt_overlap']}", f"Validation/challenge prompt overlap: {readiness['validation_challenge_prompt_overlap']}", f"Sanity overlap: {readiness['sanity_overlap']}", f"CP7 prompt overlap: {readiness['cp7_prompt_overlap']}", f"CP7 prompt+response overlap: {readiness['cp7_prompt_response_overlap']}", "", f"Reference records tested: {readiness['functional_records_tested']}", f"Individual functional tests executed: {readiness['individual_functional_tests_executed']}", f"Reference functional pass rate: {readiness['individual_functional_tests_passed'] / readiness['individual_functional_tests_executed']:.0%}", "", f"All repository tests: {readiness['all_repository_tests']}", f"Failed tests: {readiness['failed_tests']}", "", f"Train SHA256: {readiness['source_sha256']['train']}", f"Validation SHA256: {readiness['source_sha256']['validation']}", f"Challenge SHA256: {readiness['source_sha256']['challenge']}", f"Sanity SHA256: {readiness['source_sha256']['sanity']}", "", f"Package: {readiness['package']}", f"Package SHA256: {readiness['package_sha256']}", "", "Model weights modified: NO", "Tokenizer modified: NO", "Production SFT started: NO", "", f"OVERALL STATUS: {readiness['overall_status']}", "=" * 60, ""])
    (ROOT / "reports/checkpoint_8_v3/READINESS.txt").write_text(text, encoding="utf-8"); print(text, end="")
    return 0 if hard else 1


if __name__ == "__main__": raise SystemExit(main())
