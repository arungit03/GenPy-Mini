"""Training-engine CLI with mandatory explicit budgets and dry-run safety."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.config import load_config
from genpy.data.io import sha256_file
from genpy.model import GenPyForCausalLM
from genpy.model.utils import count_parameters
from genpy.training.config import load_training_config
from genpy.training.dataset import MemmapTokenDataset
from genpy.training.trainer import TrainingEngine


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--model-config", default="configs/model_200m.yaml"); parser.add_argument("--train-config", default="configs/training_engine.yaml"); parser.add_argument("--data", default="data/tokenized/genpy-32k/TOKEN_CACHE_MANIFEST.json"); parser.add_argument("--run-dir", default="runs/genpy200m_pretrain_v1"); parser.add_argument("--resume", default=None, choices=["auto"]); parser.add_argument("--session-steps", type=int, default=None); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    model_config_path, training_config_path = ROOT / args.model_config, ROOT / args.train_config
    model_config, training_config = load_config(model_config_path), load_training_config(training_config_path)
    manifest_path = ROOT / args.data; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); cache_root = manifest_path.parent
    if manifest.get("source_dataset_hash") != "17ba25f0154d1ffa04fdd4b91a22123a0770fe6aa76416ba57e4630264cb0b44": raise RuntimeError("wrong production source dataset hash")
    if sha256_file(cache_root / "train.bin") != manifest["train_bin_sha256"] or sha256_file(cache_root / "validation.bin") != manifest["validation_bin_sha256"]: raise RuntimeError("token cache integrity failure")
    model = GenPyForCausalLM(model_config.model)
    train_dataset = MemmapTokenDataset(cache_root / "train.bin", training_config.training.sequence_length)
    validation_dataset = MemmapTokenDataset(cache_root / "validation.bin", training_config.training.sequence_length)
    print(f"Model: {sum(parameter.numel() for parameter in model.parameters()):,} parameters\nDevice: {training_config.training.device}\nPrecision: {training_config.training.precision}\nTokens/update: {training_config.training.sequence_length * training_config.training.micro_batch_size * training_config.training.gradient_accumulation_steps}")
    if args.dry_run:
        training_config.validate(require_budget=False); print("Dry run: no optimizer step performed."); return 0
    training_config.validate(require_budget=True)
    if args.session_steps is not None and args.session_steps <= 0: raise ValueError("--session-steps must be positive")
    if count_parameters(model) != 201560832: raise RuntimeError("ABORT: production parameter count mismatch")
    if training_config.training.device != "cuda" or training_config.training.precision != "bf16": raise RuntimeError("ABORT: production training requires device=cuda and precision=bf16")
    engine = TrainingEngine(model, train_dataset, validation_dataset, training_config, ROOT / args.run_dir, model_config_hash=sha256_file(model_config_path), cache_hash=sha256_file(manifest_path), run_id=Path(args.run_dir).name)
    if args.resume is not None: engine.resume(args.resume)
    try: state = engine.run(args.session_steps)
    except FloatingPointError as error:
        diagnostic = engine.record_failure(error)
        print(f"Training aborted on non-finite state; diagnostic: {diagnostic}; latest known-good checkpoint preserved.")
        return 1
    except KeyboardInterrupt:
        engine.save_checkpoint(); print("Interrupted; latest checkpoint saved."); return 130
    print(f"{engine.last_run_status} | step {state.global_step}/{training_config.training.max_steps} | tokens {state.tokens_seen} | latest checkpoint: {engine.checkpoints.latest()}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
