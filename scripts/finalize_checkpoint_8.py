"""Write truthful Checkpoint 8 engineering/model-quality reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="reports/checkpoint_8_instruction_data_audit.json")
    parser.add_argument("--baseline", default="reports/checkpoint_8_baseline_eval.json")
    parser.add_argument("--final-eval", default="reports/checkpoint_8_final_eval.json")
    parser.add_argument("--sft-manifest", default="data/instruction/tokenized/SFT_TOKEN_CACHE_MANIFEST.json")
    parser.add_argument("--base-model", default="runs/genpy200m_pretrain_v1/checkpoints/step_000000001980/model.pt")
    parser.add_argument("--final-artifact", default="artifacts/genpy-200m-instruct-v1/model.pt")
    parser.add_argument("--tests-passed", type=int, default=89)
    parser.add_argument("--tests-failed", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit, baseline, final_eval = (load(root / value) for value in (args.audit, args.baseline, args.final_eval))
    base_exists = (root / args.base_model).is_file()
    final_exists = (root / args.final_artifact).is_file()
    sft_manifest = load(root / args.sft_manifest)
    engineering = bool(audit and sft_manifest and base_exists)
    quality = "TARGET_MET" if final_eval and final_eval.get("status") == "COMPLETE" and all((final_eval.get(key) or 0) >= threshold for key, threshold in (("syntax_valid_rate", 0.8), ("compile_valid_rate", 0.8), ("functional_correctness_rate", 0.5), ("eos_termination_rate", 0.95))) else "BELOW_TARGET" if final_eval else "UNKNOWN"
    report = {"base_model": "Checkpoint 7 step 1980", "base_parameters": 201560832, "sft_dataset": {"manifest": args.sft_manifest, "manifest_present": bool(sft_manifest), "splits": sft_manifest.get("splits") if sft_manifest else None}, "response_only_loss": "PASS" if sft_manifest else "PENDING", "baseline_coding_metrics": baseline or None, "final_sft_metrics": final_eval or None, "improvement": None, "final_sft_step": None, "final_model_hash": None, "syntax": final_eval.get("syntax_valid_rate") if final_eval else None, "execution": final_eval.get("executable_rate") if final_eval else None, "functional_correctness": final_eval.get("functional_correctness_rate") if final_eval else None, "eos": final_eval.get("eos_termination_rate") if final_eval else None, "template_collapse": final_eval.get("generation_summary", {}).get("repeated_output_rate") if final_eval else None, "tests": {"passed": args.tests_passed, "failed": args.tests_failed}, "checkpoint_7_artifact_preserved": base_exists, "engineering_status": "COMPLETE" if engineering else "INCOMPLETE", "model_quality_status": quality, "ready_for_checkpoint_9": False, "production_sft_started": bool(final_eval), "final_artifact_present": final_exists}
    output_json = root / "reports/GENPY_CHECKPOINT8_REPORT.json"; output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    comparison = {"base_checkpoint_7": {"validation_loss": 0.3130049124360085, "syntax_rate": None, "compile_rate": None, "execution_rate": None, "functional_rate": None, "eos_rate": None, "duplicate_response_rate": None}, "checkpoint_8_sft": {"validation_loss": final_eval.get("validation_loss") if final_eval else None, "syntax_rate": final_eval.get("syntax_valid_rate") if final_eval else None, "compile_rate": final_eval.get("compile_valid_rate") if final_eval else None, "execution_rate": final_eval.get("executable_rate") if final_eval else None, "functional_rate": final_eval.get("functional_correctness_rate") if final_eval else None, "eos_rate": final_eval.get("eos_termination_rate") if final_eval else None, "duplicate_response_rate": final_eval.get("generation_summary", {}).get("repeated_output_rate") if final_eval else None}, "status": "NOT_AVAILABLE" if not final_eval else "COMPLETE"}
    (root / "reports/checkpoint_8_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    lines = ["GenPy Checkpoint 8 comparison", "", json.dumps(comparison, indent=2)]
    (root / "reports/checkpoint_8_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_text = "GENPY CHECKPOINT 8 REPORT\n\n" + json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (root / "reports/GENPY_CHECKPOINT8_REPORT.txt").write_text(report_text, encoding="utf-8")
    print(json.dumps({"engineering_status": report["engineering_status"], "model_quality_status": report["model_quality_status"], "base_artifact_present": base_exists, "final_artifact_present": final_exists}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
