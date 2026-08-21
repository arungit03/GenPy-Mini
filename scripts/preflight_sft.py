"""Production SFT preflight; never performs an optimizer update."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.config import load_config
from genpy.data.io import sha256_file
from genpy.model import GenPyForCausalLM
from genpy.model.utils import count_parameters
from genpy.training.config import load_training_config

EXPECTED_PARAMETERS = 201560832


def hash_matches(path: Path, expected: str | None) -> bool:
    return bool(path.is_file() and expected and sha256_file(path) == expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_200m.yaml")
    parser.add_argument("--train-config", default="configs/sft_200m_kaggle.yaml")
    parser.add_argument("--data", default="data/instruction/tokenized/SFT_TOKEN_CACHE_MANIFEST.json")
    parser.add_argument("--base-model", default="runs/genpy200m_pretrain_v1/checkpoints/step_000000001980/model.pt")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--run-dir", default="runs/genpy200m_sft_v1")
    parser.add_argument("--base-sha256", default=None)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()
    model_config_path, config_path, manifest_path, base_path, tokenizer_path = map(lambda value: ROOT / value, (args.model_config, args.train_config, args.data, args.base_model, args.tokenizer))
    model_config = load_config(model_config_path)
    config = load_training_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = GenPyForCausalLM(model_config.model)
    parameters = count_parameters(model)
    cuda = torch.cuda.is_available()
    cache_root = manifest_path.parent
    train_meta = manifest.get("splits", {}).get("train", {})
    validation_meta = manifest.get("splits", {}).get("validation", {})
    base_hash = sha256_file(base_path) if base_path.is_file() else None
    hash_record = ROOT / "reports/checkpoint_7_base_model_hash.txt"
    expected_base_hash = args.base_sha256 or (hash_record.read_text(encoding="utf-8").split()[0] if hash_record.is_file() else None)
    tokenizer_manifest = tokenizer_path / "TOKENIZER_MANIFEST.json"
    tokenizer_hash = sha256_file(tokenizer_manifest) if tokenizer_manifest.is_file() else None
    train_source = str(train_meta.get("source", "")).lower()
    validation_source = str(validation_meta.get("source", "")).lower()
    mask_valid = False
    if (cache_root / "train.labels.bin").is_file():
        labels = np.memmap(cache_root / "train.labels.bin", dtype=np.int32, mode="r")
        mask_valid = bool(len(labels) and np.any(labels == -100) and np.any(labels >= 0))
    effective_positions = config.training.sequence_length * config.training.micro_batch_size * config.training.gradient_accumulation_steps
    checks = {
        "cuda_available": cuda,
        "cuda_gpu_present": bool(cuda and torch.cuda.get_device_name(0)),
        "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported()),
        "model_parameters": parameters == EXPECTED_PARAMETERS,
        "base_checkpoint_exists": base_path.is_file(),
        "base_checkpoint_hash_valid": bool(base_hash and expected_base_hash and base_hash == expected_base_hash),
        "tokenizer_hash_valid": bool(tokenizer_hash and tokenizer_hash == manifest.get("tokenizer_manifest_sha256")),
        "sft_manifest_valid": manifest.get("tokenizer_vocab_size") == 32000 and manifest.get("sequence_length") == config.training.sequence_length,
        "train_hash_valid": hash_matches(cache_root / "train.input_ids.bin", train_meta.get("input_ids_sha256")) and hash_matches(cache_root / "train.labels.bin", train_meta.get("labels_sha256")),
        "validation_hash_valid": hash_matches(cache_root / "validation.input_ids.bin", validation_meta.get("input_ids_sha256")) and hash_matches(cache_root / "validation.labels.bin", validation_meta.get("labels_sha256")),
        "no_test_data_in_training": "test" not in train_source and "test" not in validation_source and manifest.get("test_split_immutable") is True,
        "sufficient_disk": shutil.disk_usage(ROOT).free > 0,
        "effective_batch": effective_positions == 2048,
        "response_only_masks_valid": mask_valid,
        "training_budget_valid": config.training.max_steps is not None and config.training.max_steps > 0 and config.training.max_tokens is None,
    }
    result = {"checks": checks, "gpu": torch.cuda.get_device_name(0) if cuda else None, "base_model_sha256": base_hash, "model_parameters": parameters, "effective_positions_per_update": effective_positions, "effective_tokens_per_update": effective_positions, "status": "PASS" if all(checks.values()) else "FAIL"}
    print(json.dumps(result, indent=2))
    print(f"CHECKPOINT 8 SFT PREFLIGHT: {result['status']}")
    if args.json_path:
        output = ROOT / args.json_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
