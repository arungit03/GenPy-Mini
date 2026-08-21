"""Analyze response-masked SFT compute and choose the shortest safe context."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def quantiles(values: list[int]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(array.min()), "mean": float(array.mean()), "p50": float(np.percentile(array, 50, method="linear")), "p90": float(np.percentile(array, 90, method="linear")), "p95": float(np.percentile(array, 95, method="linear")), "p99": float(np.percentile(array, 99, method="linear")), "max": float(array.max())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/instruction/tokenized")
    parser.add_argument("--manifest", default="data/instruction/tokenized/SFT_TOKEN_CACHE_MANIFEST.json")
    parser.add_argument("--sequence-candidates", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--output", default="reports/checkpoint_8_sft_budget.json")
    args = parser.parse_args()
    cache = ROOT / args.cache
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    train = manifest["splits"]["train"]
    offsets = np.load(cache / "train.offsets.npy", mmap_mode="r")
    labels = np.memmap(cache / "train.labels.bin", dtype=np.int32, mode="r")
    lengths = np.diff(offsets).astype(np.int64).tolist()
    target_lengths = [int(np.count_nonzero(labels[int(offsets[index]):int(offsets[index + 1])] != -100)) for index in range(len(lengths))]
    ignored_lengths = [length - target for length, target in zip(lengths, target_lengths)]
    sequence_analysis = {}
    for sequence_length in args.sequence_candidates:
        nonpadding = [min(sequence_length, max(0, length - 1)) for length in lengths]
        padding = [sequence_length - value for value in nonpadding]
        supervised = [min(target, sequence_length) for target in target_lengths]
        sequence_analysis[str(sequence_length)] = {"truncation_count_estimate": sum(length > sequence_length + 1 for length in lengths), "total_padding_tokens": int(sum(padding)), "mean_nonpadding_tokens_per_example": statistics.fmean(nonpadding), "mean_supervised_tokens_per_example": statistics.fmean(supervised), "estimated_nonpadding_tokens_per_update": statistics.fmean(nonpadding) * 8, "estimated_supervised_tokens_per_update": statistics.fmean(supervised) * 8, "processed_padded_positions_per_pass": len(lengths) * sequence_length, "processed_nonpadding_positions_per_pass": int(sum(nonpadding))}
    safe = [sequence for sequence in args.sequence_candidates if sequence_analysis[str(sequence)]["truncation_count_estimate"] == 0]
    recommended = min(safe) if safe else max(args.sequence_candidates)
    report = {"training_examples": len(lengths), "total_formatted_tokens": int(sum(lengths)), "total_assistant_supervised_tokens": int(sum(target_lengths)), "total_ignored_prompt_tokens": int(sum(ignored_lengths)), "formatted_sequence_length": quantiles(lengths), "assistant_target_length": quantiles(target_lengths), "sequence_analysis": sequence_analysis, "recommended_sequence_length": recommended, "recommended_sequence_length_reason": "shortest candidate with zero estimated truncation; 1024 is not justified by the observed maximum length", "effective_sequences_per_update": 8, "updates_per_90000_example_pass": (len(lengths) + 7) // 8, "proposed_dataset_equivalent_passes": 1, "proposed_global_sft_steps": (len(lengths) + 7) // 8, "padding_definition": "sequence_length minus non-padding autoregressive input positions per example"}
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
