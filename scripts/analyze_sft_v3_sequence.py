"""Select v3.1 sequence length from train and validation only."""

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
from scripts.v31_common import REPORT_DIR, V3_RAW, V3_SFT, read_jsonl, write_json_and_text


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)] if ordered else 0


def distribution(values):
    return {"count": len(values), "min": min(values), "mean": statistics.mean(values), "p50": percentile(values, .50), "p90": percentile(values, .90), "p95": percentile(values, .95), "p99": percentile(values, .99), "max": max(values)}


def measure(rows, tokenizer, sequence_length=None):
    encodings = [encode_sft_record(row, tokenizer, sequence_length or 1024) for row in rows]
    return {"formatted_token_length": distribution([len(value.input_ids) for value in encodings]), "prompt_token_length": distribution([value.prompt_tokens for value in encodings]), "assistant_supervised_token_length": distribution([value.assistant_tokens for value in encodings]), "encodings": encodings}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="reports/checkpoint_8_v3_1/sequence_length_analysis.json"); args = parser.parse_args()
    tokenizer = GenPyTokenizer.load(ROOT / "artifacts/tokenizer/genpy-32k")
    rows = {name: read_jsonl(V3_SFT / f"{name}.jsonl") for name in ("train", "validation", "challenge")}
    rows["sanity"] = read_jsonl(V3_RAW / "sanity.jsonl")
    base = {name: measure(rows[name], tokenizer) for name in rows}
    candidates = {}
    for candidate in (128, 160, 192, 224, 256, 320, 384, 512):
        per_split = {}
        for name in ("train", "validation"):
            encoding = measure(rows[name], tokenizer, candidate)["encodings"]
            padding = len(rows[name]) * candidate - sum(max(0, len(value.input_ids) - 1) for value in encoding)
            per_split[name] = {"truncation_count": sum(value.truncated for value in encoding), "padding_positions": padding, "padding_fraction": padding / (len(rows[name]) * candidate)}
        candidates[str(candidate)] = {"train": per_split["train"], "validation": per_split["validation"], "combined_truncation_count": per_split["train"]["truncation_count"] + per_split["validation"]["truncation_count"], "padding_positions": per_split["train"]["padding_positions"] + per_split["validation"]["padding_positions"], "padding_fraction": (per_split["train"]["padding_positions"] + per_split["validation"]["padding_positions"]) / ((len(rows["train"]) + len(rows["validation"])) * candidate)}
    selected = next((candidate for candidate in (128, 160, 192, 224, 256, 320, 384, 512) if candidates[str(candidate)]["train"]["truncation_count"] == 0 and candidates[str(candidate)]["validation"]["truncation_count"] == 0), None)
    if selected is None: raise RuntimeError("no candidate sequence length has zero train and validation truncation")
    diagnostic = {}
    for name in ("challenge", "sanity"):
        encoding = measure(rows[name], tokenizer, selected)["encodings"]
        lengths = [len(value.input_ids) for value in encoding]
        diagnostic[name] = {"formatted_token_length": distribution(lengths), "would_truncate_at_selected_length": sum(value.truncated for value in encoding), "used_for_sequence_selection": False}
    result = {"format_version": 1, "selection_sources": ["train", "validation"], "challenge_used_for_sequence_selection": False, "sanity_used_for_sequence_selection": False, "train": {key: value for key, value in base["train"].items() if key != "encodings"}, "validation": {key: value for key, value in base["validation"].items() if key != "encodings"}, "candidates": candidates, "selected_sequence_length": selected, "challenge_diagnostic": diagnostic["challenge"], "sanity_diagnostic": diagnostic["sanity"], "selection_rule": "shortest candidate with zero train and validation truncation; frozen evaluation splits cannot alter selection"}
    write_json_and_text(ROOT / args.output, ROOT / args.output.replace(".json", ".txt"), "GenPy Checkpoint 8-v3.1 sequence analysis", result)
    print(json.dumps({"selected_sequence_length": selected, "train_truncation": candidates[str(selected)]["train"]["truncation_count"], "validation_truncation": candidates[str(selected)]["validation"]["truncation_count"], "challenge_diagnostic_truncation": diagnostic["challenge"]["would_truncate_at_selected_length"], "sanity_diagnostic_truncation": diagnostic["sanity"]["would_truncate_at_selected_length"]}, indent=2))


if __name__ == "__main__": main()
