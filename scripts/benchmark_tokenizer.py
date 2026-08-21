"""Benchmark token counts without treating compression as a complete quality score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.data.io import iter_records
from genpy.tokenizer.tokenizer import GenPyTokenizer
from genpy.tokenizer.validation import measure_texts


def row_texts(path: Path, limit: int | None = None) -> list[str]:
    texts = []
    for index, record in enumerate(iter_records(path)):
        if limit is not None and index >= limit:
            break
        instruction = str(record.get("instruction", ""))
        input_text = str(record.get("input", ""))
        response = str(record.get("response", ""))
        texts.append(instruction + ("\n" + input_text if input_text else "") + "\n" + response)
    return texts


def field_texts(path: Path, field: str, limit: int | None = None) -> list[str]:
    texts = []
    for index, record in enumerate(iter_records(path)):
        if limit is not None and index >= limit:
            break
        value = str(record.get(field, ""))
        if value:
            texts.append(value)
    return texts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--train", default="data/instruction/python/train.jsonl")
    parser.add_argument("--validation", default="data/instruction/python/validation.jsonl")
    parser.add_argument("--test", default="data/instruction/python/test.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    tokenizer = GenPyTokenizer.load(ROOT / args.tokenizer)
    reports = {}
    train_path = ROOT / args.train
    groups = {
        "python_code": field_texts(train_path, "response", args.limit),
        "natural_language_instructions": field_texts(train_path, "instruction", args.limit),
        "mixed_instruction_code": row_texts(train_path, args.limit),
    }
    for name, texts in groups.items():
        metric = measure_texts(tokenizer, texts).to_dict()
        reports[name] = metric
        print(f"{name}: {metric['characters']:,} chars, {metric['tokens']:,} tokens, "
              f"{metric['tokens_per_character']:.6f} tokens/char, {metric['unknown_tokens']} UNK ({metric['unknown_rate']:.6%})")
    for name, path in (("train", args.train), ("validation", args.validation), ("test", args.test)):
        metric = measure_texts(tokenizer, row_texts(ROOT / path, args.limit)).to_dict()
        reports[name] = metric
        print(f"{name.title()}: {metric['characters']:,} chars, {metric['bytes']:,} bytes, {metric['tokens']:,} tokens, "
              f"{metric['tokens_per_character']:.6f} tokens/char, {metric['characters_per_token']:.4f} chars/token, "
              f"{metric['bytes_per_token']:.4f} bytes/token, {metric['unknown_tokens']} UNK ({metric['unknown_rate']:.6%})")
    output = ROOT / "reports/tokenizer_benchmark.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
