"""Inspect packed tensor metadata without decoding corpus content."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.training.packed_dataset import PackedDataset  # noqa: E402
from genpy.training.packing import load_packing_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--family", choices=("pretraining", "instruction"), default="pretraining")
    args = parser.parse_args()
    config = load_packing_config(args.config, ROOT)
    dataset = PackedDataset(
        config.output_root / "manifests/packing_manifest.json",
        family=args.family,
        split=args.split,
        tokenizer_fingerprint=str(config.tokenizer["fingerprint"]),
        packing_configuration_hash=config.config_hash,
    )
    sample = dataset[0]
    checksum = hashlib.sha256(sample.input_ids.numpy().tobytes()).hexdigest()
    report = {
        "input_shape": list(sample.input_ids.shape),
        "labels_shape": list(sample.labels.shape),
        "input_dtype": str(sample.input_ids.dtype),
        "minimum_token_id": int(sample.input_ids.min()),
        "maximum_token_id": int(sample.input_ids.max()),
        "padding_positions": int(sample.attention_mask.logical_not().sum()),
        "active_labels": int(sample.labels.ne(-100).sum()),
        "tokenizer_fingerprint": config.tokenizer["fingerprint"],
        "split": args.split,
        "family": args.family,
        "shard_id": sample.shard_id,
        "sample_index": sample.sample_index,
        "input_checksum": checksum,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
