"""Analyze v3.1 response-supervised SFT token and optimizer budgets."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genpy.tokenizer import GenPyTokenizer
from genpy.training.sft_dataset import encode_sft_record
from scripts.v31_common import REPORT_DIR, TOKENIZER_DIR, V3_SFT, read_jsonl, write_json_and_text


def percentile(values, fraction):
    values = sorted(values); return values[int((len(values) - 1) * fraction)]


def distribution(values):
    return {"min": min(values), "mean": statistics.mean(values), "p50": percentile(values, .50), "p90": percentile(values, .90), "p95": percentile(values, .95), "p99": percentile(values, .99), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="reports/checkpoint_8_v3_1/sft_budget.json"); args = parser.parse_args()
    cache = json.loads((ROOT / "data/instruction/tokenized_v3/SFT_V3_TOKEN_CACHE_MANIFEST.json").read_text(encoding="utf-8")); sequence = int(cache["selected_sequence_length"]); rows = read_jsonl(V3_SFT / "train.jsonl"); tokenizer = GenPyTokenizer.load(TOKENIZER_DIR)
    encodings = [encode_sft_record(row, tokenizer, sequence) for row in rows]
    formatted = [len(value.input_ids) for value in encodings]; assistant = [value.assistant_tokens for value in encodings]
    total_formatted = sum(formatted); total_assistant = sum(assistant); total_ignored = total_formatted - total_assistant; micro, accumulation = 1, 8; examples_update = micro * accumulation; updates_pass = (len(rows) + examples_update - 1) // examples_update; actual_nonpadding_pass = total_formatted - len(rows)
    budgets = {}
    for passes in range(1, 7):
        steps = updates_pass * passes
        budgets[str(passes)] = {"dataset_equivalent_passes": passes, "optimizer_steps": steps, "approx_supervised_tokens_processed": total_assistant * passes, "approx_total_static_positions_processed": steps * sequence * examples_update, "approx_actual_non_padding_positions_processed": actual_nonpadding_pass * passes}
    result = {"format_version": 1, "train_examples": len(rows), "sequence_length": sequence, "micro_batch_size": micro, "gradient_accumulation_steps": accumulation, "effective_examples_per_update": examples_update, "total_formatted_train_tokens": total_formatted, "total_assistant_supervised_tokens": total_assistant, "total_ignored_prompt_tokens": total_ignored, "assistant_tokens_per_example": distribution(assistant), "formatted_tokens_per_example": distribution(formatted), "estimated_non_padding_token_positions_per_update": actual_nonpadding_pass / len(rows) * examples_update, "estimated_assistant_supervised_tokens_per_update": total_assistant / len(rows) * examples_update, "static_padded_positions_per_update": sequence * examples_update, "updates_per_dataset_pass": updates_pass, "candidate_global_budgets": budgets, "recommended_dataset_passes": 3, "recommended_global_max_steps": budgets["3"]["optimizer_steps"], "planned_observation_gates": [100, 250, 500], "scheduler_horizon_is_global": True, "reason": "v3 is smaller and semantically cleaner than v2; use functional evaluation gates while retaining a fixed three-pass scheduler horizon", "production_sft_started": False}
    write_json_and_text(ROOT / args.output, ROOT / args.output.replace(".json", ".txt"), "GenPy Checkpoint 8-v3.1 SFT budget", result)
    print(json.dumps({key: result[key] for key in ("total_formatted_train_tokens", "total_assistant_supervised_tokens", "total_ignored_prompt_tokens", "estimated_non_padding_token_positions_per_update", "estimated_assistant_supervised_tokens_per_update", "static_padded_positions_per_update", "updates_per_dataset_pass", "recommended_global_max_steps")}, indent=2))


if __name__ == "__main__": main()
