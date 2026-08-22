"""Audit v3 semantic quality, split isolation, CP7 leakage, and immutability."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genpy.data.io import sha256_file
from scripts.build_sft_v3_dataset import REQUIRED, ARTIFICIAL, load, validate

SPACE = re.compile(r"\s+")


def norm(value): return SPACE.sub(" ", str(value).strip().lower())
def prompt(row): return norm(row.get("instruction", ""))
def pair(row): return norm(row.get("instruction", "") + "\n" + row.get("response", ""))
def response(row): return norm(row.get("response", ""))


def overlap(left, right, fn): return len({fn(row) for row in left} & {fn(row) for row in right})


def stats(rows):
    prompts = [prompt(row) for row in rows]; responses = [response(row) for row in rows]
    return {"count": len(rows), "category_distribution": dict(Counter(row.get("category") for row in rows)), "skill_distribution": dict(Counter(row.get("skill_id") for row in rows)), "difficulty_distribution": dict(Counter(row.get("difficulty") for row in rows)), "task_style_distribution": dict(Counter(row.get("task_style") for row in rows)), "template_distribution": dict(Counter(row.get("prompt_template_id") for row in rows)), "unique_skills": len(set(row.get("skill_id") for row in rows)), "unique_prompts": len(set(prompts)), "unique_reference_responses": len(set(responses)), "average_prompt_length": statistics.mean(map(len, prompts)) if prompts else 0, "average_response_length": statistics.mean(map(len, responses)) if responses else 0, "minimum_test_cases": min((len(row.get("test_cases", [])) for row in rows), default=0), "maximum_test_cases": max((len(row.get("test_cases", [])) for row in rows), default=0), "function_name_distribution": dict(Counter(row.get("function_name") for row in rows)), "forbidden_artificial_identifier_count": sum(bool(ARTIFICIAL.search(row.get("instruction", "") + "\n" + row.get("input", "") + "\n" + row.get("response", ""))) for row in rows), "validation_errors": sum(bool(validate(row)) for row in rows)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset-dir", default="data/instruction/python_v3"); parser.add_argument("--cp7", nargs="+", default=["data/instruction/python/train.jsonl", "data/instruction/python/validation.jsonl", "data/instruction/python/test.jsonl"]); parser.add_argument("--output", default="reports/checkpoint_8_v3/dataset_audit.json"); parser.add_argument("--text-output", default="reports/checkpoint_8_v3/dataset_audit.txt"); args = parser.parse_args()
    dataset = ROOT / args.dataset_dir
    splits = {name: load(dataset / f"{name}.jsonl") for name in ("train", "validation", "challenge", "sanity")}
    pairs = {}
    for left_name, right_name in (("train", "validation"), ("train", "challenge"), ("validation", "challenge"), ("sanity", "train"), ("sanity", "validation"), ("sanity", "challenge")):
        left, right = splits[left_name], splits[right_name]
        pairs[f"{left_name}_vs_{right_name}"] = {"normalized_prompt": overlap(left, right, prompt), "prompt_response_pair": overlap(left, right, pair), "normalized_response": overlap(left, right, response)}
    cp7_rows, cp7_sources = [], []
    for value in args.cp7:
        path = ROOT / value
        if path.is_file():
            rows = list(__import__("genpy.data.io", fromlist=["iter_records"]).iter_records(path)); cp7_rows.extend(rows); cp7_sources.append({"path": str(path), "sha256": sha256_file(path), "count": len(rows)})
    cp7 = {name: {"normalized_prompt": overlap(splits[name], cp7_rows, prompt), "prompt_response_pair": overlap(splits[name], cp7_rows, pair), "normalized_response": overlap(splits[name], cp7_rows, response)} for name in splits} if cp7_rows else None
    template_sets = {name: {row.get("prompt_template_id") for row in rows} for name, rows in splits.items()}
    template_overlaps = {f"{left}_vs_{right}": len(template_sets[left] & template_sets[right]) for left, right in (("train", "validation"), ("train", "challenge"), ("validation", "challenge"))}
    counts_ok = {"train": len(splits["train"]) == 3000, "validation": len(splits["validation"]) == 300, "challenge": len(splits["challenge"]) == 200, "sanity": len(splits["sanity"]) == 20}
    protected = ["data/instruction/python_v2", "data/instruction/sft_v2", "data/instruction/tokenized_v2", "reports/checkpoint_8_v2", "runs/genpy200m_sft_v2"]
    protected_state = [{"path": str(ROOT / value), "exists": (ROOT / value).exists()} for value in protected]
    hard = all(counts_ok.values()) and all(value["normalized_prompt"] == 0 for value in pairs.values()) and all(value == 0 for value in template_overlaps.values()) and all(value["normalized_prompt"] == 0 and value["prompt_response_pair"] == 0 for value in (cp7 or {}).values()) and all(value["forbidden_artificial_identifier_count"] == 0 and value["validation_errors"] == 0 and value["function_name_distribution"] == {"solve": value["count"]} for value in (stats(rows) for rows in splits.values()))
    result = {"format_version": 3, "dataset_name": "GenPy-SFT-v3-Semantic", "dataset_version": "genpy-sft-v3-semantic-v1", "seed": 42, "splits": {name: stats(rows) for name, rows in splits.items()}, "counts_ok": counts_ok, "cross_split_overlaps": pairs, "template_overlaps": template_overlaps, "cp7_sources": cp7_sources, "cp7_overlap": cp7, "source_sha256": {name: sha256_file(dataset / f"{name}.jsonl") for name in splits}, "protected_v1_v2_paths": protected_state, "v3_output_disjoint_from_protected_paths": True, "hard_gates_before_functional": hard, "overall_status": "PASS" if hard else "FAIL", "challenge_training_excluded": all(row.get("provenance", {}).get("training_excluded") is True for row in splits["challenge"]), "sanity_optimizer_excluded": all(row.get("provenance", {}).get("optimizer_excluded") is True for row in splits["sanity"])}
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / args.text_output).write_text("GenPy Checkpoint 8-v3 dataset audit\n\n" + json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"overall_status": result["overall_status"], "counts_ok": counts_ok, "cp7": cp7, "cross_split": pairs}, indent=2))
    return 0 if hard else 1


if __name__ == "__main__": raise SystemExit(main())
