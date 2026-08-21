"""Build an explicit, audited SFT JSONL view without touching source splits."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.data.io import iter_records, sha256_file


def process_split(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = destination.with_name(destination.stem + ".rejected.jsonl")
    counts = {"seen": 0, "accepted": 0, "rejected": 0, "rejection_reasons": {}}
    with destination.open("w", encoding="utf-8", newline="\n") as accepted, rejected_path.open("w", encoding="utf-8", newline="\n") as rejected:
        for record in iter_records(source):
            counts["seen"] += 1
            reason = None
            if not str(record.get("instruction", "")).strip():
                reason = "empty_instruction"
            elif not str(record.get("response", "")).strip():
                reason = "empty_response"
            else:
                try:
                    ast.parse(str(record["response"]))
                    compile(str(record["response"]), "<sft-build>", "exec")
                except (SyntaxError, ValueError, TypeError):
                    reason = "syntax_invalid"
            if reason:
                counts["rejected"] += 1
                counts["rejection_reasons"][reason] = counts["rejection_reasons"].get(reason, 0) + 1
                rejected.write(json.dumps({"reason": reason, "record": record}, ensure_ascii=False, sort_keys=True) + "\n")
            else:
                counts["accepted"] += 1
                accepted.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {"source": str(source), "source_sha256": sha256_file(source), "output": str(destination), "output_sha256": sha256_file(destination), "rejected_output": str(rejected_path), **counts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/instruction/python/train.jsonl")
    parser.add_argument("--validation", default="data/instruction/python/validation.jsonl")
    parser.add_argument("--test", default="data/instruction/python/test.jsonl")
    parser.add_argument("--output-dir", default="data/instruction/sft")
    parser.add_argument("--manifest", default="data/instruction/sft/SFT_DATASET_MANIFEST.json")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    splits = {"train": ROOT / args.train, "validation": ROOT / args.validation, "test": ROOT / args.test}
    stats = {name: process_split(source, output_dir / f"{name}.jsonl") for name, source in splits.items()}
    manifest = {"format_version": 1, "dataset_name": "GenPy-Python-100K-SFT", "source": "genpy_programmatic_v1", "test_split_immutable": True, "splits": stats, "tokenizer": "GenPy-Tokenizer-32K", "execution_correctness_claimed": False}
    manifest_path = ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in value.items() if key in {"seen", "accepted", "rejected"}} for name, value in stats.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
