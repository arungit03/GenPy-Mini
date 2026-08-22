"""Run every v2 reference function in an isolated subprocess."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genpy.data.io import iter_records
from genpy.evaluation.coding import run_function_tests_subprocess


def run_one(row: dict) -> dict:
    result = run_function_tests_subprocess(str(row["response"]), str(row["function_name"]), row["test_cases"], timeout_seconds=2.0)
    return {"id": row["id"], "passed": bool(result.get("passed")), "category": result.get("category"), "test_count": len(row["test_cases"])}


def audit(path: Path, workers: int) -> dict:
    rows = list(iter_records(path))
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda value: value["id"])
    failures = [value for value in results if not value["passed"]]
    return {"path": str(path), "count": len(rows), "tested": len(results), "passed": len(results) - len(failures), "failed": len(failures), "functional_rate": (len(results) - len(failures)) / len(results) if results else 0, "failure_categories": dict(Counter(value["category"] for value in failures)), "failures": failures[:100], "minimum_test_cases": min((value["test_count"] for value in results), default=0), "all_pass": not failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/instruction/python_v2")
    parser.add_argument("--output", default="reports/checkpoint_8_v2/reference_functional_audit.json")
    parser.add_argument("--text-output", default="reports/checkpoint_8_v2/reference_functional_audit.txt")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    dataset_dir = ROOT / args.dataset_dir
    stats = {name: audit(dataset_dir / f"{name}.jsonl", args.workers) for name in ("train", "validation", "challenge", "sanity")}
    result = {"format_version": 2, "execution": "isolated subprocess per reference response", "timeout_seconds": 2.0, "safe_scope": "no risky imports; temporary cwd; restricted environment", "splits": stats, "all_pass": all(value["all_pass"] for value in stats.values()), "total": sum(value["count"] for value in stats.values()), "passed": sum(value["passed"] for value in stats.values()), "failed": sum(value["failed"] for value in stats.values())}
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / args.text_output).write_text("GenPy Checkpoint 8-v2 functional audit\n\n" + json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": result["all_pass"], "total": result["total"], "passed": result["passed"], "failed": result["failed"], "by_split": {name: value["functional_rate"] for name, value in stats.items()}}, indent=2))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
