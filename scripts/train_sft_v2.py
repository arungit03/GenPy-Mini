"""Checkpoint 8-v2 SFT entry point; production execution is externally gated."""

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

EXPECTED_BASE_SHA256 = "a963a91d8f6bee350e15ff88d3375c039887cb0b09c787fecf0f2de02d5be942"
EXPECTED_PARAMETERS = 201560832


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_200m.yaml")
    parser.add_argument("--train-config", default="configs/sft_200m_kaggle_v2.yaml")
    parser.add_argument("--data", default="data/instruction/tokenized_v2/SFT_V2_TOKEN_CACHE_MANIFEST.json")
    parser.add_argument("--base-model", default="runs/genpy200m_pretrain_v1/checkpoints/step_000000001980/model.pt")
    parser.add_argument("--run-dir", default="runs/genpy200m_sft_v2")
    parser.add_argument("--resume", default=None, choices=["auto"])
    parser.add_argument("--session-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any("sft_v1" in str(value).lower() or "tokenized" in str(value).lower() and "tokenized_v2" not in str(value).lower() for value in (args.train_config, args.data, args.run_dir)):
        raise RuntimeError("ABORT: v2 entry point received a v1 path")
    if args.session_steps is not None and args.session_steps <= 0:
        raise ValueError("--session-steps must be positive")
    config = load_training_config(ROOT / args.train_config)
    manifest_path = ROOT / args.data
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config.training.max_steps != 2500 or config.training.sampling != "shuffled_epoch":
        raise RuntimeError("ABORT: v2 scheduler budget/sampler is not the finalized configuration")
    if "challenge" not in manifest.get("splits", {}) or "sanity" in manifest.get("splits", {}):
        raise RuntimeError("ABORT: v2 cache manifest must include challenge metadata but exclude sanity from optimizer cache")
    train_meta = manifest["splits"]["train"]; validation_meta = manifest["splits"]["validation"]
    if "challenge" in str(train_meta.get("source", "")).lower() or "sanity" in str(train_meta.get("source", "")).lower():
        raise RuntimeError("ABORT: challenge/sanity cannot be SFT training data")
    model_config = load_config(ROOT / args.model_config)
    model = GenPyForCausalLM(model_config.model)
    parameters = count_parameters(model)
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"ABORT: parameter count {parameters} != {EXPECTED_PARAMETERS}")
    base_path = ROOT / args.base_model
    if base_path.name != "model.pt":
        raise RuntimeError("ABORT: Checkpoint 7 trust target must be model.pt, never a tar archive")
    if not base_path.is_file():
        print(f"Checkpoint 7 model.pt: UNAVAILABLE ({base_path})")
        if args.dry_run:
            print("Dry run: no optimizer step performed; SFT remains unstarted.")
            return 0
        raise FileNotFoundError(base_path)
    base_hash = sha256_file(base_path)
    if base_hash != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"ABORT: Checkpoint 7 model.pt SHA256 {base_hash} != trusted {EXPECTED_BASE_SHA256}")
    if args.dry_run:
        print(f"Checkpoint 7 model.pt SHA256: {base_hash}\nDry run: no optimizer step performed.")
        return 0
    if config.training.device != "cuda" or config.training.precision != "bf16":
        raise RuntimeError("ABORT: production v2 SFT requires cuda/bf16")
    model.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=False))
    cache_root = manifest_path.parent
    train_dataset = SFTMemmapDataset(cache_root / "train.input_ids.bin", cache_root / "train.labels.bin", cache_root / "train.offsets.npy", config.training.sequence_length)
    validation_dataset = SFTMemmapDataset(cache_root / "validation.input_ids.bin", cache_root / "validation.labels.bin", cache_root / "validation.offsets.npy", config.training.sequence_length)
    engine = SFTTrainingEngine(model, train_dataset, validation_dataset, config, ROOT / args.run_dir, base_hash, sha256_file(manifest_path))
    if args.resume == "auto":
        engine.resume("auto")
    state = engine.run(args.session_steps)
    print(f"{engine.last_run_status} | step {state.global_step}/{config.training.max_steps} | supervised tokens {state.tokens_seen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
