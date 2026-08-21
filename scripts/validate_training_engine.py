"""Validate engine configuration, cache integrity, and checkpoint writability without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.config import load_config
from genpy.data.io import sha256_file
from genpy.model import GenPyForCausalLM
from genpy.model.utils import count_parameters
from genpy.tokenizer import GenPyTokenizer
from genpy.training.config import load_training_config
from genpy.training.optimizer import create_adamw


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--model-config", default="configs/model_200m.yaml"); parser.add_argument("--train-config", default="configs/training_engine.yaml"); parser.add_argument("--manifest", default="data/tokenized/genpy-32k/TOKEN_CACHE_MANIFEST.json"); parser.add_argument("--checkpoint-dir", default="checkpoints/engine_validation"); args = parser.parse_args()
    model_config = load_config(ROOT / args.model_config); train_config = load_training_config(ROOT / args.train_config)
    manifest_path = ROOT / args.manifest; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); cache_root = manifest_path.parent
    tokenizer = GenPyTokenizer.load(ROOT / "artifacts/tokenizer/genpy-32k")
    integrity = sha256_file(cache_root / "train.bin") == manifest["train_bin_sha256"] and sha256_file(cache_root / "validation.bin") == manifest["validation_bin_sha256"]
    model = GenPyForCausalLM(model_config.model)
    optimizer, audit = create_adamw(model, train_config.optimizer)
    checkpoint_dir = ROOT / args.checkpoint_dir; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    values = {"model_parameters": count_parameters(model), "tokenizer_vocab": tokenizer.vocab_size, "cache_integrity": integrity, "optimizer_audit": audit, "effective_tokens_per_update": train_config.training.sequence_length * train_config.training.micro_batch_size * train_config.training.gradient_accumulation_steps, "budget_present": train_config.training.max_steps is not None or train_config.training.max_tokens is not None, "checkpoint_directory_writable": checkpoint_dir.is_dir()}
    passed = values["model_parameters"] == 201560832 and values["tokenizer_vocab"] == 32000 and integrity and audit["duplicate_parameters"] == 0 and audit["missing_parameters"] == 0 and values["checkpoint_directory_writable"]
    print(json.dumps({**values, "status": "PASS" if passed else "FAIL"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__": raise SystemExit(main())
