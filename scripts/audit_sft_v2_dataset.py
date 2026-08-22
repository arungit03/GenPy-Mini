"""Audit v2 novelty, split isolation, and contamination against CP7 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genpy.data.io import iter_records, sha256_file

SPACE = re.compile(r"\s+")
NUMBER = re.compile(r"\d+")


def norm(value: object) -> str:
    return SPACE.sub(" ", str(value).strip().lower())


def record_key(row: dict) -> str:
    return norm(json.dumps({key: row.get(key) for key in ("instruction", "input", "response", "function_name", "test_cases")}, sort_keys=True, ensure_ascii=False))


def prompt_key(row: dict) -> str:
    return norm(row.get("instruction", ""))


def prompt_input_key(row: dict) -> str:
    return norm(str(row.get("instruction", "")) + "\n" + str(row.get("input", "")))


def response_key(row: dict) -> str:
    return norm(row.get("response", ""))


def template_key(row: dict) -> str:
    value = norm(row.get("instruction", ""))
    value = NUMBER.sub("<NUM>", value)
    value = re.sub(r"solve_[a-z0-9_]+_(?:train|validation|challenge)_\d+", "<FUNCTION>", value)
    return value


def load(path: Path) -> list[dict]:
    return list(iter_records(path))


def overlap(left: list[dict], right: list[dict], fn) -> int:
    return len({fn(row) for row in left} & {fn(row) for row in right})


def summarize(rows: list[dict]) -> dict:
    return {
        "count": len(rows), "categories": dict(Counter(row.get("category") for row in rows)),
        "skills": len({row.get("skill_id") for row in rows}), "templates": len({row.get("prompt_template_id") for row in rows}),
        "task_styles": dict(Counter(row.get("task_style") for row in rows)), "difficulties": dict(Counter(row.get("difficulty") for row in rows)),
        "function_names": len({row.get("function_name") for row in rows}), "test_case_counts": dict(Counter(len(row.get("test_cases", [])) for row in rows)),
        "response_unique_rate": len({response_key(row) for row in rows}) / len(rows) if rows else 0,
        "top_responses": Counter(response_key(row) for row in rows).most_common(10),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/instruction/python_v2")
    parser.add_argument("--cp7", nargs="+", default=["data/instruction/python/train.jsonl", "data/instruction/python/validation.jsonl", "data/instruction/python/test.jsonl"])
    parser.add_argument("--output", default="reports/checkpoint_8_v2/dataset_audit.json")
    parser.add_argument("--text-output", default="reports/checkpoint_8_v2/dataset_audit.txt")
    args = parser.parse_args()
    dataset_dir = ROOT / args.dataset_dir
    splits = {name: load(dataset_dir / f"{name}.jsonl") for name in ("train", "validation", "challenge", "sanity")}
    cp7_rows, cp7_sources = [], []
    for value in args.cp7:
        path = ROOT / value
        if not path.exists():
            continue
        rows = load(path); cp7_rows.extend(rows)
        cp7_sources.append({"path": str(path), "sha256": sha256_file(path), "count": len(rows)})
    pairs = {}
    for left_name, right_name in (("train", "validation"), ("train", "challenge"), ("validation", "challenge"), ("train", "sanity"), ("validation", "sanity"), ("challenge", "sanity")):
        left, right = splits[left_name], splits[right_name]
        pairs[f"{left_name}_vs_{right_name}"] = {
            "record": overlap(left, right, record_key), "prompt": overlap(left, right, prompt_key),
            "instruction_input": overlap(left, right, prompt_input_key), "response": overlap(left, right, response_key),
            "template_id": overlap(left, right, lambda row: row.get("prompt_template_id")), "template": overlap(left, right, template_key),
        }
    cp7_checks = {"available": bool(cp7_rows), "source_count": len(cp7_sources)}
    for name in ("train", "validation", "challenge", "sanity"):
        cp7_checks[name] = {"record": overlap(splits[name], cp7_rows, record_key), "prompt": overlap(splits[name], cp7_rows, prompt_key), "instruction_input": overlap(splits[name], cp7_rows, prompt_input_key), "response": overlap(splits[name], cp7_rows, response_key), "template": overlap(splits[name], cp7_rows, template_key)} if cp7_rows else None
    challenge_template_disjoint = not ({row.get("prompt_template_id") for row in splits["train"]} & {row.get("prompt_template_id") for row in splits["challenge"]})
    v2_hashes = {name: sha256_file(dataset_dir / f"{name}.jsonl") for name in ("train", "validation", "challenge", "sanity")}
    source_hashes_differ = bool(cp7_sources) and all(v2_hashes["train"] != source["sha256"] and v2_hashes["validation"] != source["sha256"] for source in cp7_sources)
    all_pass = bool(cp7_rows) and source_hashes_differ and challenge_template_disjoint and all(value["record"] == value["prompt"] == value["instruction_input"] == value["response"] == value["template_id"] == value["template"] == 0 for value in pairs.values()) and all(value["record"] == value["prompt"] == value["instruction_input"] == value["response"] == value["template"] == 0 for value in cp7_checks.values() if isinstance(value, dict))
    result = {"format_version": 2, "splits": {name: summarize(rows) for name, rows in splits.items()}, "source_hashes": v2_hashes, "source_hashes_differ_from_cp7": source_hashes_differ, "split_overlaps": pairs, "cp7_sources": cp7_sources, "cp7_contamination": cp7_checks, "train_challenge_template_ids_disjoint": challenge_template_disjoint, "status": "PASS" if all_pass else "FAIL", "challenge_status": "CLEAN_GENERALIZATION" if all_pass else "CONTAMINATED_OR_UNPROVEN", "policy": {"challenge_training": False, "challenge_hyperparameter_selection": False, "sanity_training": False, "sanity_validation": False}}
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / args.text_output).write_text("GenPy Checkpoint 8-v2 dataset audit\n\n" + json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "challenge_status": result["challenge_status"], "cp7_available": bool(cp7_rows), "train_challenge": pairs["train_vs_challenge"]}, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
