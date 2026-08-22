"""Validate and materialize the immutable v2 SFT views."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.data.io import iter_records, sha256_file

REQUIRED = {"id", "instruction", "input", "response", "response_type", "category", "skill_id", "prompt_template_id", "difficulty", "task_style", "function_name", "test_cases", "provenance"}
FORBIDDEN_IMPORTS = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests", "urllib", "http", "ftplib"}


def validate_record(record: dict) -> list[str]:
    errors = []
    missing = REQUIRED - record.keys()
    if missing:
        errors.append("missing:" + ",".join(sorted(missing)))
    if record.get("response_type") != "code":
        errors.append("response_type")
    if not str(record.get("instruction", "")).strip() or not str(record.get("response", "")).strip():
        errors.append("empty_text")
    cases = record.get("test_cases")
    if not isinstance(cases, list) or len(cases) < 3:
        errors.append("test_cases")
    else:
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("args", []), list) or not isinstance(case.get("kwargs", {}), dict) or "expected" not in case:
                errors.append("test_case_schema")
                break
    try:
        tree = ast.parse(str(record.get("response", "")))
        compile(tree, "<v2-reference>", "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name.split(".")[0] in FORBIDDEN_IMPORTS for alias in node.names):
                errors.append("unsafe_import")
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in FORBIDDEN_IMPORTS:
                errors.append("unsafe_import")
            if isinstance(node, (ast.Call,)) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "compile", "__import__"}:
                errors.append("dynamic_execution")
    except (SyntaxError, ValueError, TypeError):
        errors.append("syntax_or_compile")
    return sorted(set(errors))


def process(source: Path, destination: Path, expected_count: int) -> dict:
    rows = list(iter_records(source))
    errors = []
    ids = set()
    for index, record in enumerate(rows, 1):
        if record.get("id") in ids:
            errors.append({"index": index, "id": record.get("id"), "errors": ["duplicate_id"]})
        ids.add(record.get("id"))
        problems = validate_record(record)
        if problems:
            errors.append({"index": index, "id": record.get("id"), "errors": problems})
    if len(rows) != expected_count:
        errors.append({"index": 0, "errors": [f"count:{len(rows)} != {expected_count}"]})
    if errors:
        raise RuntimeError(f"{source} failed v2 validation with {len(errors)} errors; no rows were discarded")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "source": str(source), "source_sha256": sha256_file(source), "output": str(destination), "output_sha256": sha256_file(destination),
        "count": len(rows), "syntax_valid": len(rows), "compile_valid": len(rows), "rejected": 0,
        "category_distribution": dict(Counter(row["category"] for row in rows)),
        "skill_count": len({row["skill_id"] for row in rows}), "template_count": len({row["prompt_template_id"] for row in rows}),
        "task_style_distribution": dict(Counter(row["task_style"] for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/instruction/python_v2")
    parser.add_argument("--output-dir", default="data/instruction/sft_v2")
    parser.add_argument("--manifest", default="data/instruction/sft_v2/SFT_V2_DATASET_MANIFEST.json")
    args = parser.parse_args()
    source_dir, output_dir = ROOT / args.source_dir, ROOT / args.output_dir
    stats = {name: process(source_dir / f"{name}.jsonl", output_dir / f"{name}.jsonl", count) for name, count in {"train": 10000, "validation": 1000, "challenge": 500}.items()}
    sanity = source_dir / "sanity.jsonl"
    sanity_count = sum(1 for _ in iter_records(sanity))
    if sanity_count < 20:
        raise RuntimeError("v2 sanity set must contain at least 20 records")
    manifest = {
        "format_version": 2, "dataset_name": "GenPy-SFT-v2-Pilot", "dataset_version": "genpy-sft-v2-pilot-v1",
        "generator_version": "genpy-sft-v2-pilot-v1", "seed": 42, "splits": stats,
        "sanity": {"source": str(sanity), "source_sha256": sha256_file(sanity), "count": sanity_count, "training_excluded": True, "validation_excluded": True},
        "challenge": {"training_excluded": True, "hyperparameter_selection_excluded": True}, "external_data_used": False,
        "functional_audit_required": True, "test_case_minimum": 3,
    }
    manifest_path = ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: value["count"] for name, value in stats.items()} | {"sanity": sanity_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
