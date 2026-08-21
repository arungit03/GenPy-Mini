"""Checkpoint 8 SFT entry point with immutable-base and fixed-budget guards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.config import load_config
from genpy.data.io import sha256_file
from genpy.model import GenPyForCausalLM
from genpy.model.utils import count_parameters
from genpy.training.config import load_training_config
from genpy.training.sft_dataset import SFTMemmapDataset
from genpy.training.sft_trainer import SFTTrainingEngine

EXPECTED_PARAMETERS = 201560832


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_200m.yaml")
    parser.add_argument("--train-config", default="configs/sft_200m_kaggle.yaml")
    parser.add_argument("--data", default="data/instruction/tokenized/SFT_TOKEN_CACHE_MANIFEST.json")
    parser.add_argument("--base-model", default="runs/genpy200m_pretrain_v1/checkpoints/step_000000001980/model.pt")
    parser.add_argument("--run-dir", default="runs/genpy200m_sft_v1")
    parser.add_argument("--resume", default=None, choices=["auto"])
    parser.add_argument("--session-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    model_config_path = ROOT / args.model_config
    train_config = load_training_config(ROOT / args.train_config)
    manifest_path = ROOT / args.data
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_root = manifest_path.parent
    model_config = load_config(model_config_path)
    model = GenPyForCausalLM(model_config.model)
    parameters = count_parameters(model)
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"ABORT: SFT model parameter count is {parameters}, expected {EXPECTED_PARAMETERS}")
    train = manifest["splits"]["train"]
    validation = manifest["splits"]["validation"]
    for name, metadata in (("train", train), ("validation", validation)):
        if sha256_file(cache_root / f"{name}.input_ids.bin") != metadata["input_ids_sha256"] or sha256_file(cache_root / f"{name}.labels.bin") != metadata["labels_sha256"]:
            raise RuntimeError(f"ABORT: {name} SFT cache hash mismatch")
    if "test" in str(train.get("source", "")).lower() or "test" in str(validation.get("source", "")).lower():
        raise RuntimeError("ABORT: test split may not be used for SFT training")
    effective_update = train_config.training.sequence_length * train_config.training.micro_batch_size * train_config.training.gradient_accumulation_steps
    updates_per_pass = (train["document_count"] + train_config.training.micro_batch_size * train_config.training.gradient_accumulation_steps - 1) // (train_config.training.micro_batch_size * train_config.training.gradient_accumulation_steps)
    print(f"Model parameters: {parameters:,}\nSFT train examples: {train['document_count']:,}\nAssistant target tokens: {train['assistant_token_count']:,}\nApprox tokens/update: {effective_update:,}\nUpdates per dataset pass: {updates_per_pass:,}")
    base_path = ROOT / args.base_model
    if not base_path.is_file():
        print(f"Base checkpoint: MISSING ({base_path})")
        if args.dry_run:
            print("Dry run: no optimizer step performed; restore the immutable Checkpoint 7 model before SFT.")
            return 0
        raise FileNotFoundError(f"immutable Checkpoint 7 model not found: {base_path}")
    base_hash = sha256_file(base_path)
    if args.dry_run:
        print(f"Base checkpoint SHA256: {base_hash}\nDry run: no optimizer step performed.")
        return 0
    if train_config.training.max_steps is None:
        raise ValueError("set a fixed training.max_steps in configs/sft_200m_kaggle.yaml after SFT budget analysis")
    if train_config.training.device != "cuda" or train_config.training.precision != "bf16":
        raise RuntimeError("ABORT: production SFT requires device=cuda and precision=bf16")
    if args.session_steps is not None and args.session_steps <= 0:
        raise ValueError("--session-steps must be positive")
    model.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=False))
    train_dataset = SFTMemmapDataset(cache_root / "train.input_ids.bin", cache_root / "train.labels.bin", cache_root / "train.offsets.npy", train_config.training.sequence_length)
    validation_dataset = SFTMemmapDataset(cache_root / "validation.input_ids.bin", cache_root / "validation.labels.bin", cache_root / "validation.offsets.npy", train_config.training.sequence_length)
    engine = SFTTrainingEngine(model, train_dataset, validation_dataset, train_config, ROOT / args.run_dir, base_hash, sha256_file(manifest_path))
    if args.resume == "auto":
        engine.resume("auto")
    try:
        state = engine.run(args.session_steps)
    except FloatingPointError as error:
        diagnostic = ROOT / args.run_dir / "sft_numerical_failure.json"
        diagnostic.write_text(json.dumps({"error": str(error), "step": engine.state.global_step, "latest_checkpoint": str(engine.checkpoints.latest()) if engine.checkpoints.latest() else None}, indent=2) + "\n", encoding="utf-8")
        print(f"SFT aborted on non-finite state; diagnostic written to {diagnostic}")
        return 1
    print(f"{engine.last_run_status} | step {state.global_step}/{train_config.training.max_steps} | supervised tokens {state.tokens_seen} | latest checkpoint {engine.checkpoints.latest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
