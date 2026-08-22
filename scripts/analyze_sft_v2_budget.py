"""Produce the CP8-v2 SFT compute/budget audit."""

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


def pct(values: list[int], f: float) -> int:
    values = sorted(values)
    return values[int((len(values) - 1) * f)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/instruction/sft_v2/train.jsonl")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--output", default="reports/checkpoint_8_v2/sft_budget.json")
    args = parser.parse_args()
    sequence = args.sequence_length or int(json.loads((ROOT / "reports/checkpoint_8_v2/sequence_length_analysis.json").read_text())["selected_sequence_length"])
    tokenizer = GenPyTokenizer.load(ROOT / args.tokenizer)
    rows = list(iter_records(ROOT / args.dataset))
    encodings = [encode_sft_record(row, tokenizer, sequence) for row in rows]
    formatted = [len(value.input_ids) for value in encodings]
    assistant = [value.assistant_tokens for value in encodings]
    total_formatted, total_assistant = sum(formatted), sum(assistant)
    total_ignored = total_formatted - total_assistant
    update_examples = args.micro_batch_size * args.gradient_accumulation
    updates_per_pass = (len(rows) + update_examples - 1) // update_examples
    actual_non_padding = total_formatted - len(rows)  # SFTMemmapDataset shifts x/y and therefore consumes one position per document less.
    result = {"format_version": 2, "examples": len(rows), "micro_batch_size": args.micro_batch_size, "gradient_accumulation": args.gradient_accumulation, "sequence_length": sequence, "total_formatted_tokens": total_formatted, "total_assistant_supervised_tokens": total_assistant, "total_ignored_prompt_tokens": total_ignored, "total_padding_tokens_if_static": len(rows) * sequence - total_formatted, "formatted_sequence_length": {"min": min(formatted), "mean": statistics.mean(formatted), "p50": pct(formatted, .50), "p90": pct(formatted, .90), "p95": pct(formatted, .95), "p99": pct(formatted, .99), "max": max(formatted)}, "assistant_target_length": {"min": min(assistant), "mean": statistics.mean(assistant), "p50": pct(assistant, .50), "p90": pct(assistant, .90), "p95": pct(assistant, .95), "p99": pct(assistant, .99), "max": max(assistant)}, "estimated_actual_non_padding_tokens_per_update": actual_non_padding / len(rows) * update_examples, "estimated_assistant_supervised_tokens_per_update": total_assistant / len(rows) * update_examples, "updates_per_pass": updates_per_pass, "proposed_dataset_equivalent_passes": args.passes, "proposed_global_sft_steps": updates_per_pass * args.passes, "estimated_processed_token_positions_per_pass": len(rows) * sequence, "estimated_actual_non_padding_token_positions_per_pass": actual_non_padding, "reason": "fixed 2 epochs over 10,000 examples; 10,000 / (1*8) = 1,250 updates per epoch and 2,500 global steps, independent of session limits", "sequence_length_evaluation": "256 is the shortest candidate with zero truncation across train, validation, and challenge; 192 truncates 10 frozen challenge examples, so it is not production-safe for the complete cache"}
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("total_formatted_tokens", "total_assistant_supervised_tokens", "total_ignored_prompt_tokens", "formatted_sequence_length", "assistant_target_length", "estimated_actual_non_padding_tokens_per_update", "estimated_assistant_supervised_tokens_per_update", "proposed_global_sft_steps")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
