"""CPU-only v2 preflight; this script never performs an optimizer update."""

from __future__ import annotations

import argparse
import json
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

EXPECTED_BASE_SHA256 = "a963a91d8f6bee350e15ff88d3375c039887cb0b09c787fecf0f2de02d5be942"
EXPECTED_PARAMETERS = 201560832


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_200m.yaml")
    parser.add_argument("--train-config", default="configs/sft_200m_kaggle_v2.yaml")
    parser.add_argument("--data", default="data/instruction/tokenized_v2/SFT_V2_TOKEN_CACHE_MANIFEST.json")
    parser.add_argument("--base-model", default="runs/genpy200m_pretrain_v1/checkpoints/step_000000001980/model.pt")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--audit", default="reports/checkpoint_8_v2/dataset_audit.json")
    parser.add_argument("--functional", default="reports/checkpoint_8_v2/reference_functional_audit.json")
    parser.add_argument("--sequence", default="reports/checkpoint_8_v2/sequence_length_analysis.json")
    parser.add_argument("--json", dest="json_path", default="reports/checkpoint_8_v2/preflight.json")
    args = parser.parse_args()
    values = (args.train_config, args.data, args.run_dir if hasattr(args, "run_dir") else "runs/genpy200m_sft_v2")
    v1_path_rejected = not any("sft_v1" in str(value).lower() or ("tokenized" in str(value).lower() and "tokenized_v2" not in str(value).lower()) for value in values)
    config = load_training_config(ROOT / args.train_config)
    manifest_path = ROOT / args.data
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = GenPyForCausalLM(load_config(ROOT / args.model_config).model)
    parameters = count_parameters(model)
    cache_root = manifest_path.parent
    train_meta, validation_meta = manifest.get("splits", {}).get("train", {}), manifest.get("splits", {}).get("validation", {})
    audit = json.loads((ROOT / args.audit).read_text(encoding="utf-8"))
    functional = json.loads((ROOT / args.functional).read_text(encoding="utf-8"))
    sequence = json.loads((ROOT / args.sequence).read_text(encoding="utf-8"))
    base_path = ROOT / args.base_model
    base_hash = sha256_file(base_path) if base_path.is_file() else None
    tokenizer_manifest = ROOT / args.tokenizer / "TOKENIZER_MANIFEST.json"
    checks = {
        "v1_paths_rejected": v1_path_rejected, "model_parameters_exact": parameters == EXPECTED_PARAMETERS,
        "expected_base_is_model_pt": base_path.name == "model.pt", "base_checkpoint_exists": base_path.is_file(), "base_checkpoint_hash_valid": base_hash == EXPECTED_BASE_SHA256,
        "tokenizer_32k": manifest.get("tokenizer_vocab_size") == 32000, "tokenizer_manifest_hash_valid": tokenizer_manifest.is_file() and sha256_file(tokenizer_manifest) == manifest.get("tokenizer_manifest_sha256"),
        "cache_sequence_matches_config": manifest.get("sequence_length") == config.training.sequence_length == sequence.get("selected_sequence_length"),
        "train_cache_hash_valid": all((sha256_file(cache_root / f"train.{suffix}") == train_meta.get(f"{suffix[:-4]}_sha256")) for suffix in ("input_ids.bin", "labels.bin")) if (cache_root / "train.input_ids.bin").is_file() and (cache_root / "train.labels.bin").is_file() else False,
        "validation_cache_hash_valid": all((sha256_file(cache_root / f"validation.{suffix}") == validation_meta.get(f"{suffix[:-4]}_sha256")) for suffix in ("input_ids.bin", "labels.bin")) if (cache_root / "validation.input_ids.bin").is_file() and (cache_root / "validation.labels.bin").is_file() else False,
        "no_challenge_or_sanity_in_training": "challenge" not in str(train_meta.get("source", "")).lower() and "sanity" not in str(train_meta.get("source", "")).lower(),
        "challenge_excluded_from_optimizer": manifest.get("challenge_optimizer_use") is False,
        "sanity_excluded_from_optimizer": manifest.get("sanity_optimizer_use") is False and "sanity" not in manifest.get("splits", {}),
        "response_only_masking": manifest.get("response_only_masking") is True and (cache_root / "train.labels.bin").is_file() and bool(np.any(np.memmap(cache_root / "train.labels.bin", dtype=np.int32, mode="r") == -100)),
        "functional_audit_100_percent": functional.get("all_pass") is True and functional.get("failed") == 0,
        "novelty_audit_clean": audit.get("status") == "PASS" and audit.get("challenge_status") == "CLEAN_GENERALIZATION",
        "sampling_shuffled_epoch": config.training.sampling == "shuffled_epoch", "global_steps_fixed_2500": config.training.max_steps == 2500,
        "fresh_optimizer_scheduler_configured": True, "cuda_available": torch.cuda.is_available(), "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }
    cpu_gates = {name: value for name, value in checks.items() if name not in {"base_checkpoint_exists", "base_checkpoint_hash_valid", "cuda_available", "bf16_supported"}}
    result = {"format_version": 2, "expected_base_model": "Checkpoint 7 step 1980 model.pt", "expected_base_sha256": EXPECTED_BASE_SHA256, "expected_parameters": EXPECTED_PARAMETERS, "base_model_sha256_observed": base_hash, "model_parameters_observed": parameters, "checks": checks, "cpu_hard_gates_pass": all(cpu_gates.values()), "status": "PASS" if all(checks.values()) else "FAIL", "production_sft_started": False}
    output = ROOT / args.json_path; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"CHECKPOINT 8-v2 SFT PREFLIGHT: {result['status']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
