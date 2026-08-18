"""Fast CPU end-to-end Step 6 training, checkpoint, and resume smoke test."""

import json
import tempfile
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import numpy as np
import torch

from genpy.config import ModelConfig, TrainingConfig
from genpy.model import GenPyForCausalLM
from genpy.training import PackedTokenDataset, TrainingEngine, create_dataloader
from genpy.training.checkpoint import CheckpointManager
from genpy.training.logger import TrainingLogger


def configs():
    model = ModelConfig("tiny-training", 64, 16, 32, 2, 4, 8, 64, 1e-5, 10000.0, True)
    train = TrainingConfig(7, 2, 2, 0.002, 0.0002, 0.01, 0.2, 1.0, "fp32", 1, 100, 2, "checkpoints", "logs", 8, 0.9, 0.95, 1e-8, 0, False, 2, 2)
    return model, train


def make_dataset(directory: Path, split: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tokens = np.arange(64, dtype=np.uint16)
    path = directory / f"{split}.bin"
    if not path.exists():
        tokens.tofile(path)
    metadata_path = directory / f"{split}_metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps({"format_version": 1, "dtype": "uint16", "split": split, "token_count": len(tokens), "document_count": 1, "vocab_size": 64, "special_token_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3}}), encoding="utf-8")
    return path


def build(directory: Path, model_config, train_config, max_steps: int):
    dataset = PackedTokenDataset(make_dataset(directory, "train"), train_config.sequence_length)
    loader, sampler = create_dataloader(dataset, train_config.micro_batch_size, seed=train_config.seed, shuffle=True)
    model = GenPyForCausalLM(model_config)
    manager = CheckpointManager(directory / "checkpoints", keep_last=2)
    engine = TrainingEngine(model, train_config, loader, device="cpu", max_steps=max_steps, checkpoint_manager=manager, logger=TrainingLogger(directory / "logs"), train_sampler=sampler)
    return engine, manager


def main() -> int:
    torch.manual_seed(123)
    model_config, train_config = configs()
    with tempfile.TemporaryDirectory(prefix="genpy-step6-smoke-") as temp:
        root = Path(temp)
        torch.manual_seed(123)
        continuous, _ = build(root / "continuous", model_config, train_config, 4)
        continuous_result = continuous.train()
        torch.manual_seed(123)
        resumed, manager = build(root / "resumed", model_config, train_config, 4)
        resumed.train(stop_after_steps=2)
        checkpoint = resumed.save_checkpoint()
        resumed2, manager2 = build(root / "resumed", model_config, train_config, 4)
        resumed2.load_checkpoint(checkpoint)
        result = resumed2.train()
        same = all(torch.equal(a, b) for a, b in zip(continuous.model.parameters(), resumed2.model.parameters()))
        print(f"Initial/reload/continue/finish: PASS")
        print(f"Initial loss: {continuous_result['initial_loss']:.6f}")
        print(f"Final loss: {continuous_result['final_loss']:.6f}")
        print(f"Steps: {result['global_step']}")
        print(f"Tokens seen: {result['tokens_seen']}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Resume state matches continuous: {same}")
        for candidate in (continuous, resumed, resumed2):
            candidate.train_loader.dataset.close()
        if not same:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
