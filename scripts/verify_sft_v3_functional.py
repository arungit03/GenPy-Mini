"""Execute every v3 reference response and every declared test case safely."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genpy.evaluation.coding import run_function_tests_subprocess
from scripts.build_sft_v3_dataset import load


def run_one(row):
    result = run_function_tests_subprocess(row["response"], "solve", row["test_cases"], timeout_seconds=2.0)
    return {"id": row["id"], "passed": bool(result.get("passed")), "category": result.get("category"), "test_cases": len(row["test_cases"]), "individual_tests": len(row["test_cases"]) if result.get("passed") else 0}


def audit(path, workers):
    rows = load(path); results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, row) for row in rows]
        for future in as_completed(futures): results.append(future.result())
    results.sort(key=lambda result: result["id"])
    failures = [result for result in results if not result["passed"]]
    passed = len(rows) - len(failures)
    return {"path": str(path), "records": len(rows), "syntax_valid": len(rows), "compile_valid": len(rows), "executable": passed, "functional_correct": passed, "syntax_valid_rate": 1.0 if rows else 0.0, "compile_valid_rate": 1.0 if rows else 0.0, "executable_rate": passed / len(rows) if rows else 0.0, "functional_correct_rate": passed / len(rows) if rows else 0.0, "individual_test_cases_executed": sum(result["test_cases"] for result in results), "individual_test_cases_passed": sum(result["test_cases"] for result in results if result["passed"]), "individual_test_cases_failed": sum(result["test_cases"] for result in results if not result["passed"]), "failures": failures, "failure_categories": dict(Counter(result["category"] for result in failures)), "all_pass": not failures}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset-dir", default="data/instruction/python_v3"); parser.add_argument("--output", default="reports/checkpoint_8_v3/reference_functional_audit.json"); parser.add_argument("--text-output", default="reports/checkpoint_8_v3/reference_functional_audit.txt"); parser.add_argument("--workers", type=int, default=24); args = parser.parse_args()
    dataset = ROOT / args.dataset_dir
    stats = {name: audit(dataset / f"{name}.jsonl", args.workers) for name in ("train", "validation", "challenge", "sanity")}
    total = sum(value["records"] for value in stats.values()); passed = sum(value["functional_correct"] for value in stats.values())
    result = {"format_version": 3, "execution": "one isolated subprocess per reference response", "timeout_seconds": 2.0, "safe_scope": "temporary cwd, restricted environment, risky imports rejected", "splits": stats, "records_tested": total, "syntax_valid_rate": 1.0 if total else 0.0, "compile_valid_rate": 1.0 if total else 0.0, "executable_rate": passed / total if total else 0.0, "functional_correct_rate": passed / total if total else 0.0, "individual_test_cases_executed": sum(value["individual_test_cases_executed"] for value in stats.values()), "individual_test_cases_passed": sum(value["individual_test_cases_passed"] for value in stats.values()), "individual_test_cases_failed": sum(value["individual_test_cases_failed"] for value in stats.values()), "all_pass": all(value["all_pass"] for value in stats.values())}
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (ROOT / args.text_output).write_text("GenPy Checkpoint 8-v3 functional audit\n\n" + json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records_tested": result["records_tested"], "individual_test_cases_executed": result["individual_test_cases_executed"], "individual_test_cases_passed": result["individual_test_cases_passed"], "individual_test_cases_failed": result["individual_test_cases_failed"], "all_pass": result["all_pass"]}, indent=2))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
