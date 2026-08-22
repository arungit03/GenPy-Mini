"""Choose the shortest v2 SFT sequence length with zero truncation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genpy.data.io import iter_records
from genpy.tokenizer import GenPyTokenizer
from genpy.training.sft_dataset import encode_sft_record


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))] if ordered else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/instruction/sft_v2")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--output", default="reports/checkpoint_8_v2/sequence_length_analysis.json")
    args = parser.parse_args()
    tokenizer = GenPyTokenizer.load(ROOT / args.tokenizer)
    dataset_path = ROOT / args.dataset
    source_paths = [dataset_path / f"{name}.jsonl" for name in ("train", "validation", "challenge")] if dataset_path.is_dir() else [dataset_path]
    records = [row for path in source_paths for row in iter_records(path)]
    lengths, assistant = [], []
    for row in records:
        shortest = encode_sft_record(row, tokenizer, 4096)
        lengths.append(len(shortest.input_ids)); assistant.append(shortest.assistant_tokens)
    candidates = {}
    for sequence_length in (128, 192, 256, 384, 512):
        truncated = sum(encode_sft_record(row, tokenizer, sequence_length).truncated for row in records)
        candidates[str(sequence_length)] = {"truncation_count": truncated, "truncation_rate": truncated / len(records), "padded_positions_per_pass": len(records) * sequence_length, "padding_positions_per_pass": len(records) * sequence_length - sum(min(sequence_length, length - 1) for length in lengths)}
    selected = next((value for value in (128, 192, 256, 384, 512) if candidates[str(value)]["truncation_count"] == 0), None)
    if selected is None:
        raise RuntimeError("no candidate sequence length has zero truncation")
    result = {"format_version": 2, "dataset": str(ROOT / args.dataset), "source_splits": [str(path) for path in source_paths], "count": len(records), "formatted_sequence_length": {"min": min(lengths), "mean": statistics.mean(lengths), "p50": percentile(lengths, .50), "p90": percentile(lengths, .90), "p95": percentile(lengths, .95), "p99": percentile(lengths, .99), "max": max(lengths)}, "assistant_target_length": {"min": min(assistant), "mean": statistics.mean(assistant), "p50": percentile(assistant, .50), "p90": percentile(assistant, .90), "p95": percentile(assistant, .95), "p99": percentile(assistant, .99), "max": max(assistant)}, "candidates": candidates, "selected_sequence_length": selected, "selection_rule": "shortest candidate with zero truncation across train, validation, and frozen challenge", "response_only_masking": True}
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_sequence_length": selected, "p95": result["formatted_sequence_length"]["p95"], "p99": result["formatted_sequence_length"]["p99"], "max": result["formatted_sequence_length"]["max"], "candidates": candidates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
