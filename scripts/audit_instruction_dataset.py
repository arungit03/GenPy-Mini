"""Audit SFT splits for quality, duplication, leakage, and syntax validity."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.data.io import iter_records, sha256_file

_SPACE = re.compile(r"\s+")
_TEMPLATE = re.compile(r"\d+")


def normalized(value: str) -> str:
    return _SPACE.sub(" ", value.strip().lower())


def template_normalized(value: str) -> str:
    return _TEMPLATE.sub("<NUM>", normalized(value))


def audit_split(path: Path) -> tuple[dict, list[dict]]:
    records = []
    stats = Counter()
    exact_keys = Counter()
    prompt_keys = Counter()
    solution_keys = Counter()
    template_keys = Counter()
    family_keys = Counter()
    for record in iter_records(path):
        stats["records_seen"] += 1
        instruction = str(record.get("instruction", ""))
        input_text = str(record.get("input", ""))
        response = str(record.get("response", ""))
        prompt = instruction + "\n" + input_text
        exact_key = json.dumps(record, sort_keys=True, ensure_ascii=False)
        exact_keys[exact_key] += 1
        prompt_keys[normalized(prompt)] += 1
        solution_keys[normalized(response)] += 1
        template_keys[template_normalized(prompt)] += 1
        family_keys[str(record.get("family_id", ""))] += 1
        if not instruction.strip() or not response.strip():
            stats["malformed_empty"] += 1
        try:
            ast.parse(response)
            compile(response, "<sft-audit>", "exec")
            stats["syntax_valid"] += 1
            stats["compile_valid"] += 1
        except (SyntaxError, ValueError, TypeError):
            stats["syntax_invalid"] += 1
        if not response.strip():
            stats["empty_responses"] += 1
        if len(response) > 20000:
            stats["very_long_responses"] += 1
        records.append({"id": record.get("id"), "prompt": normalized(prompt), "solution": normalized(response), "template": template_normalized(prompt), "family": str(record.get("family_id", ""))})
    stats["exact_duplicate_records"] = sum(count - 1 for count in exact_keys.values() if count > 1)
    stats["prompt_duplicate_records"] = sum(count - 1 for count in prompt_keys.values() if count > 1)
    stats["solution_duplicate_records"] = sum(count - 1 for count in solution_keys.values() if count > 1)
    stats["highly_similar_template_records"] = sum(count - 1 for count in template_keys.values() if count > 1)
    stats["unique_families"] = len(family_keys)
    return {"path": str(path), "sha256": sha256_file(path), **dict(stats)}, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/instruction/python/train.jsonl")
    parser.add_argument("--validation", default="data/instruction/python/validation.jsonl")
    parser.add_argument("--test", default="data/instruction/python/test.jsonl")
    parser.add_argument("--output", default="reports/checkpoint_8_instruction_data_audit.json")
    parser.add_argument("--text-output", default="reports/checkpoint_8_instruction_data_audit.txt")
    args = parser.parse_args()
    paths = {"train": ROOT / args.train, "validation": ROOT / args.validation, "test": ROOT / args.test}
    split_stats, split_records = {}, {}
    for name, path in paths.items():
        split_stats[name], split_records[name] = audit_split(path)
    sets = {name: {item["prompt"] for item in records} for name, records in split_records.items()}
    solutions = {name: {item["solution"] for item in records} for name, records in split_records.items()}
    templates = {name: {item["template"] for item in records} for name, records in split_records.items()}
    families = {name: {item["family"] for item in records} for name, records in split_records.items()}
    intersections = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        intersections[f"{left}_{right}_prompt_overlap"] = len(sets[left] & sets[right])
        intersections[f"{left}_{right}_solution_overlap"] = len(solutions[left] & solutions[right])
        intersections[f"{left}_{right}_template_overlap"] = len(templates[left] & templates[right])
        intersections[f"{left}_{right}_family_overlap"] = len(families[left] & families[right])
    warnings = []
    if intersections["train_test_prompt_overlap"] or intersections["train_test_solution_overlap"]:
        warnings.append("direct normalized train/test prompt or solution overlap detected")
    if intersections["train_test_template_overlap"]:
        warnings.append("train/test programmatic template-family overlap detected; inspect family-level separation")
    if split_stats["train"].get("syntax_invalid", 0) or split_stats["validation"].get("syntax_invalid", 0):
        warnings.append("syntax-invalid records exist in train or validation")
    legacy_contamination = bool(intersections["train_test_solution_overlap"] or intersections["train_test_template_overlap"])
    result = {"format_version": 1, "dataset_name": "GenPy-Python-100K", "source": "genpy_programmatic_v1", "splits": split_stats, "cross_split": intersections, "warnings": warnings, "test_split_immutable": True, "test_loaded_for_audit_only": True, "execution_correctness_claimed": False, "evaluation_classification": {"validation": "DEVELOPMENT_IN_DISTRIBUTION", "original_production_test": "LEGACY_IN_DISTRIBUTION_CONTAMINATION_RISK" if legacy_contamination else "LEGACY_TEST_NOT_CLEAN_GENERALIZATION", "clean_challenge": "DEFERRED_TO_FROZEN_CHALLENGE_AUDIT"}}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["GenPy Checkpoint 8 instruction data audit", "", json.dumps(result, indent=2, ensure_ascii=False)]
    (ROOT / args.text_output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"train": split_stats["train"], "validation": split_stats["validation"], "test": split_stats["test"], "warnings": warnings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
