"""Validate and materialize the v3 SFT views without discarding records."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"id", "instruction", "input", "response", "response_type", "category", "skill_id", "prompt_template_id", "difficulty", "task_style", "function_name", "test_cases", "provenance"}
RISKY = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests", "urllib", "http", "ftplib"}
ARTIFICIAL = re.compile(r"(?:_train_|_validation_|_challenge_|Scenario\s+\d+|\b(?:task|example\s+id)\s+\d+)", re.I)


def validate(row):
    errors = []
    missing = REQUIRED - row.keys()
    if missing: errors.append("missing_schema:" + ",".join(sorted(missing)))
    if row.get("function_name") != "solve": errors.append("function_name")
    if row.get("response_type") != "code": errors.append("response_type")
    if row.get("difficulty") not in {"easy", "medium"}: errors.append("difficulty")
    if ARTIFICIAL.search(str(row.get("instruction", "")) + "\n" + str(row.get("input", "")) + "\n" + str(row.get("response", ""))): errors.append("artificial_identifier")
    tests = row.get("test_cases")
    if not isinstance(tests, list) or len(tests) < 5: errors.append("minimum_test_cases")
    else:
        for case in tests:
            if not isinstance(case, dict) or not isinstance(case.get("args"), list) or not isinstance(case.get("kwargs"), dict) or "expected" not in case:
                errors.append("test_case_schema"); break
    try:
        tree = ast.parse(str(row.get("response", "")))
        compile(tree, "<v3>", "exec")
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1 or functions[0].name != "solve": errors.append("response_function_name")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name.split(".")[0] in RISKY for alias in node.names): errors.append("unsafe_import")
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in RISKY: errors.append("unsafe_import")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "compile", "__import__"}: errors.append("dynamic_execution")
    except (SyntaxError, ValueError, TypeError): errors.append("syntax_or_compile")
    return sorted(set(errors))


def load(path):
    with path.open(encoding="utf-8") as handle: return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--source-dir", default="data/instruction/python_v3"); parser.add_argument("--output-dir", default="data/instruction/sft_v3"); args = parser.parse_args()
    source, output = ROOT / args.source_dir, ROOT / args.output_dir
    expected = {"train": 3000, "validation": 300, "challenge": 200}
    stats, all_errors = {}, []
    for name, count in expected.items():
        rows = load(source / f"{name}.jsonl"); ids = set(); errors = []
        if len(rows) != count: errors.append({"errors": [f"count:{len(rows)} != {count}"]})
        for index, row in enumerate(rows):
            if row.get("id") in ids: errors.append({"index": index, "id": row.get("id"), "errors": ["duplicate_id"]})
            ids.add(row.get("id")); problems = validate(row)
            if problems: errors.append({"index": index, "id": row.get("id"), "errors": problems})
        if errors: all_errors.extend([{**error, "split": name} for error in errors])
        destination = output / f"{name}.jsonl"; destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows: handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        stats[name] = {"count": len(rows), "accepted": len(rows) if not errors else 0, "rejected": 0 if not errors else len(errors), "source": str(source / f"{name}.jsonl"), "source_sha256": sha(source / f"{name}.jsonl"), "output": str(destination), "output_sha256": sha(destination), "category_distribution": dict(Counter(row.get("category") for row in rows)), "skill_distribution": dict(Counter(row.get("skill_id") for row in rows)), "difficulty_distribution": dict(Counter(row.get("difficulty") for row in rows)), "task_style_distribution": dict(Counter(row.get("task_style") for row in rows)), "template_distribution": dict(Counter(row.get("prompt_template_id") for row in rows))}
    sanity = load(source / "sanity.jsonl")
    if len(sanity) != 20: all_errors.append({"split": "sanity", "errors": [f"count:{len(sanity)} != 20"]})
    sanity_errors = [validate(row) for row in sanity]
    if any(sanity_errors): all_errors.append({"split": "sanity", "errors": sanity_errors})
    manifest = {"format_version": 3, "dataset_name": "GenPy-SFT-v3-Semantic", "dataset_version": "genpy-sft-v3-semantic-v1", "seed": 42, "splits": stats, "sanity": {"source": str(source / "sanity.jsonl"), "source_sha256": sha(source / "sanity.jsonl"), "count": len(sanity), "optimizer_excluded": True, "validation_excluded": True}, "challenge": {"optimizer_excluded": True, "hyperparameter_selection_excluded": True}, "rejected_records": len(all_errors), "tokenization_performed": False, "training_performed": False}
    manifest_path = output / "SFT_V3_DATASET_MANIFEST.json"; manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if all_errors: raise RuntimeError(f"v3 validation failed with {len(all_errors)} errors; no records were discarded")
    print(json.dumps({"train": stats["train"]["count"], "validation": stats["validation"]["count"], "challenge": stats["challenge"]["count"], "sanity": len(sanity), "rejected": 0}, indent=2))


if __name__ == "__main__": main()
