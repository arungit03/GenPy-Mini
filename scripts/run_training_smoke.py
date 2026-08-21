"""Tiny local engine smoke and exact interrupted/resumed determinism test."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from _verification_cli import ROOT
from genpy.config import load_config
from genpy.model import GenPyForCausalLM
from genpy.training.config import load_training_config
from genpy.training.dataset import MemmapTokenDataset
from genpy.training.trainer import TrainingEngine
from genpy.utils.reproducibility import set_seed


def make_cache(root: Path) -> tuple[Path, Path]:
    train = np.arange(512, dtype=np.uint16) % 128
    validation = (np.arange(256, dtype=np.uint16) + 3) % 128
    train_path, validation_path = root / "train.bin", root / "validation.bin"
    train.tofile(train_path); validation.tofile(validation_path)
    return train_path, validation_path


def make_engine(model_config, engine_config, train_path, validation_path, run_dir, max_steps):
    tiny_model_config = replace(model_config, vocab_size=128, max_seq_len=32, n_layers=2, d_model=64, n_heads=4, head_dim=16, ffn_hidden_size=128)
    engine_config = replace(engine_config, training=replace(engine_config.training, max_steps=max_steps, sequence_length=32), validation=replace(engine_config.validation, batches=2))
    set_seed(engine_config.training.seed, deterministic=True)
    model = GenPyForCausalLM(tiny_model_config, attention_backend="eager")
    return TrainingEngine(model, MemmapTokenDataset(train_path, 32), MemmapTokenDataset(validation_path, 32), engine_config, run_dir, run_id="training_smoke")


def equal_state(first, second) -> bool:
    return all(torch.equal(first[key], second[key]) for key in first)


def equal_nested(first, second) -> bool:
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(equal_nested(first[key], second[key]) for key in first)
    if isinstance(first, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(equal_nested(a, b) for a, b in zip(first, second))
    return first == second


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--production", action="store_true"); parser.add_argument("--model-config", default="configs/model_200m.yaml"); parser.add_argument("--train-config", default="configs/training_smoke.yaml"); args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="genpy_training_smoke_") as temp:
        root = Path(temp); train_path, validation_path = make_cache(root); base = load_config(ROOT / "configs/model_200m.yaml"); smoke_config = load_training_config(ROOT / args.train_config)
        # Local smoke intentionally uses a tiny architecture; --production remains a bounded integration option.
        if args.production:
            print("Production smoke requested; no long run is started.")
        first = make_engine(base.model, smoke_config, train_path, validation_path, root / "run_a", 6)
        metrics_a = [first.train_optimizer_step() for _ in range(6)]; losses_a = [metric["train_loss"] for metric in metrics_a]
        first.save_checkpoint()
        state_a, model_a, optim_a, sched_a = first.state, copy.deepcopy(first.model.state_dict()), copy.deepcopy(first.optimizer.state_dict()), copy.deepcopy(first.scheduler.state_dict())
        second = make_engine(base.model, smoke_config, train_path, validation_path, root / "run_b", 6)
        losses_b_pre = [second.train_optimizer_step()["train_loss"] for _ in range(3)]; second.save_checkpoint()
        resumed = make_engine(base.model, smoke_config, train_path, validation_path, root / "run_b", 6); resumed.resume("auto")
        losses_b_post = [resumed.train_optimizer_step()["train_loss"] for _ in range(3)]
        model_equal = equal_state(model_a, resumed.model.state_dict()); optimizer_equal = equal_nested(resumed.optimizer.state_dict(), optim_a); scheduler_equal = equal_nested(resumed.scheduler.state_dict(), sched_a)
        result = {"model": "tiny GenPy architecture", "parameters": sum(p.numel() for p in first.model.parameters()), "device": str(first.device), "precision": first.precision.mode, "steps": 6, "sequence_length": 32, "micro_batch_size": 1, "accumulation": 1, "initial_loss": losses_a[0], "final_loss": losses_a[-1], "all_losses": losses_a, "validation_loss": first.validate(), "gradient_norms": [metric["gradient_norm"] for metric in metrics_a], "tokens_seen": state_a.tokens_seen, "checkpoint_saved": first.checkpoints.latest() is not None, "resume_tested": True, "exact_resume": {"model_state": model_equal, "optimizer_state": optimizer_equal, "scheduler_state": scheduler_equal, "loss_continuation": losses_a[3:] == losses_b_post, "passed": model_equal and optimizer_equal and scheduler_equal and losses_a[3:] == losses_b_post}}
        for engine in (first, second, resumed):
            engine.train_dataset.close(); engine.validation_dataset.close()
        (ROOT / "reports").mkdir(exist_ok=True); (ROOT / "reports/checkpoint_6_smoke_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Smoke: {'PASS' if result['exact_resume']['passed'] and result['checkpoint_saved'] else 'FAIL'}; loss {losses_a[0]:.4f} -> {losses_a[-1]:.4f}; exact resume {result['exact_resume']['passed']}")
        return 0 if result["exact_resume"]["passed"] and result["checkpoint_saved"] else 1


if __name__ == "__main__": raise SystemExit(main())
