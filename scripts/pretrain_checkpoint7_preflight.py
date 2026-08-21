"""Read-only production preflight for the Checkpoint 7 Kaggle run."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.config import load_config
from genpy.data.io import sha256_file
from genpy.model import GenPyForCausalLM
from genpy.model.utils import count_parameters
from genpy.training.config import load_training_config

EXPECTED_SOURCE_HASH = "17ba25f0154d1ffa04fdd4b91a22123a0770fe6aa76416ba57e4630264cb0b44"
EXPECTED_PARAMETERS = 201560832


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_200m.yaml")
    parser.add_argument("--train-config", default="configs/pretrain_200m_kaggle.yaml")
    parser.add_argument("--data", default="data/tokenized/genpy-32k/TOKEN_CACHE_MANIFEST.json")
    parser.add_argument("--run-dir", default="runs/genpy200m_pretrain_v1")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    model_path = ROOT / args.model_config
    train_path = ROOT / args.train_config
    manifest_path = ROOT / args.data
    model_config = load_config(model_path)
    train_config = load_training_config(train_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_root = manifest_path.parent
    model = GenPyForCausalLM(model_config.model)
    model_parameters = count_parameters(model)
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
    checkpoint_dir = ROOT / args.run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=checkpoint_dir, prefix=".preflight-", delete=True):
            checkpoint_writable = True
    except OSError:
        checkpoint_writable = False
    disk_free = shutil.disk_usage(ROOT).free
    t = train_config.training
    o = train_config.optimizer
    s = train_config.scheduler
    checks = {
        "cuda_available": cuda_available,
        "bf16_supported": bf16_supported,
        "model_parameters": model_parameters == EXPECTED_PARAMETERS,
        "model_config_hash_present": bool(sha256_file(model_path)),
        "source_dataset_hash": manifest.get("source_dataset_hash") == EXPECTED_SOURCE_HASH,
        "train_bin_hash": sha256_file(cache_root / "train.bin") == manifest.get("train_bin_sha256"),
        "validation_bin_hash": sha256_file(cache_root / "validation.bin") == manifest.get("validation_bin_sha256"),
        "vocab_size": manifest.get("vocab_size") == 32000 and model_config.model.vocab_size == 32000,
        "bos_token_id": manifest.get("bos_token_id") == 1,
        "eos_token_id": manifest.get("eos_token_id") == 2,
        "sequence_length": t.sequence_length == 1024,
        "micro_batch_size": t.micro_batch_size == 1,
        "gradient_accumulation_steps": t.gradient_accumulation_steps == 8,
        "effective_tokens_per_update": t.sequence_length * t.micro_batch_size * t.gradient_accumulation_steps == 8192,
        "adamw": o.name.lower() == "adamw" and o.learning_rate == 3.0e-4 and o.weight_decay == 0.1 and o.beta1 == 0.9 and o.beta2 == 0.95 and o.eps == 1.0e-8,
        "scheduler": s.type.lower() == "cosine" and s.warmup_steps == 100 and s.minimum_learning_rate == 3.0e-5,
        "checkpoint_directory_writable": checkpoint_writable,
        "available_disk_space": disk_free > 0,
        "production_budget": t.max_steps == 1980 and t.max_tokens is None,
        "production_precision_config": t.device == "cuda" and t.precision == "bf16",
    }
    result = {"checks": checks, "cuda_available": cuda_available, "gpu_name": gpu_name, "bf16_supported": bf16_supported, "model_parameters": model_parameters, "model_config_hash": sha256_file(model_path), "train_bin_sha256": sha256_file(cache_root / "train.bin"), "validation_bin_sha256": sha256_file(cache_root / "validation.bin"), "available_disk_bytes": disk_free, "effective_tokens_per_update": 8192, "production_budget_steps": 1980, "status": "PASS" if all(checks.values()) else "FAIL"}
    for key, value in result.items():
        if key != "checks": print(f"{key}: {value}")
    for key, value in checks.items(): print(f"{key}: {'PASS' if value else 'FAIL'}")
    print(f"CHECKPOINT 7 PREFLIGHT: {result['status']}")
    if args.json_path:
        output = ROOT / args.json_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
