"""Prove that the frozen challenge prompts/solutions are absent from GenPy splits."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.data.io import iter_records, sha256_file

SPACE = re.compile(r"\s+")
NUMBER = re.compile(r"\d+")


def norm(value: str) -> str:
    return SPACE.sub(" ", value.strip().lower())


def template(value: str) -> str:
    return NUMBER.sub("<NUM>", norm(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", default="data/manifests/checkpoint_8_challenge.json")
    parser.add_argument("--sources", nargs="+", default=["data/instruction/python/train.jsonl", "data/instruction/python/validation.jsonl", "data/instruction/python/test.jsonl"])
    parser.add_argument("--output", default="reports/checkpoint_8_challenge_contamination.json")
    parser.add_argument("--text-output", default="reports/checkpoint_8_challenge_contamination.txt")
    args = parser.parse_args()
    challenge_path = ROOT / args.challenge
    challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
    challenge_prompts = {norm(item["instruction"]) for item in challenge["examples"]}
    challenge_solutions = {norm(item["response"]) for item in challenge["examples"]}
    challenge_templates = {template(item["instruction"]) for item in challenge["examples"]}
    overlaps = {"prompt": [], "solution": [], "template": []}
    sources = []
    for source_value in args.sources:
        source = ROOT / source_value
        source_info = {"path": str(source), "sha256": sha256_file(source), "records": 0}
        for record in iter_records(source):
            source_info["records"] += 1
            prompt, solution = norm(str(record.get("instruction", ""))), norm(str(record.get("response", "")))
            if prompt in challenge_prompts:
                overlaps["prompt"].append({"source": str(source), "id": record.get("id")})
            if solution in challenge_solutions:
                overlaps["solution"].append({"source": str(source), "id": record.get("id")})
            if template(str(record.get("instruction", ""))) in challenge_templates:
                overlaps["template"].append({"source": str(source), "id": record.get("id")})
        sources.append(source_info)
    clean = not any(overlaps.values())
    result = {"format_version": 1, "challenge_manifest": str(challenge_path), "challenge_manifest_sha256": sha256_file(challenge_path), "challenge_name": challenge["name"], "provenance": challenge["provenance"], "sft_training_use": challenge["sft_training_use"], "hyperparameter_selection_use": challenge["hyperparameter_selection_use"], "sources_checked": sources, "overlaps": {key: {"count": len(value), "examples": value[:20]} for key, value in overlaps.items()}, "status": "CLEAN_GENERALIZATION" if clean else "CONTAMINATED_OR_UNPROVEN"}
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (ROOT / args.text_output).write_text("GenPy frozen challenge contamination audit\n\n" + json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "prompt_overlap": len(overlaps["prompt"]), "solution_overlap": len(overlaps["solution"]), "template_overlap": len(overlaps["template"])}, indent=2))
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
