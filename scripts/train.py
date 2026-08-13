"""Explicit-max-steps GenPy training CLI."""

import argparse
import json
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from genpy.config import load_model_config, load_training_config
from genpy.model import GenPyForCausalLM
from genpy.training import PackedTokenDataset, PrecisionManager, TrainingEngine, create_dataloader
from genpy.training.checkpoint import CheckpointManager
from genpy.training.logger import TrainingLogger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_200m.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--resume", nargs="?", const="latest", help="exact interrupted-run continuation")
    mode_group.add_argument("--init-from-checkpoint", type=Path, help="initialize a fresh phase from model weights only")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    model_config = load_model_config(args.model_config)
    train_config = load_training_config(args.train_config)
    if train_config.sequence_length > model_config.max_seq_len:
        raise ValueError("training sequence length exceeds model context")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive and explicit")
    if args.device.startswith("cuda") and not __import__("torch").cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if args.precision:
        train_config = type(train_config)(**{**train_config.__dict__, "precision": args.precision})
    model = GenPyForCausalLM(model_config)
    precision = PrecisionManager(train_config.precision, args.device)
    train_dataset = PackedTokenDataset(args.train_data, train_config.sequence_length)
    train_loader, train_sampler = create_dataloader(train_dataset, train_config.micro_batch_size, seed=train_config.seed, shuffle=True, num_workers=train_config.num_workers, pin_memory=train_config.pin_memory)
    validation_loader = None
    if args.validation_data:
        validation_dataset = PackedTokenDataset(args.validation_data, train_config.sequence_length)
        validation_loader, _ = create_dataloader(validation_dataset, train_config.micro_batch_size, seed=train_config.seed, shuffle=False, num_workers=train_config.num_workers, pin_memory=train_config.pin_memory)
    print(json.dumps({
        "model": model_config.name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": args.device,
        "precision": args.precision or train_config.precision,
        "sequence_length": train_config.sequence_length,
        "micro_batch_size": train_config.micro_batch_size,
        "gradient_accumulation": train_config.gradient_accumulation_steps,
        "effective_sequences_per_update": train_config.micro_batch_size * train_config.gradient_accumulation_steps,
        "effective_tokens_per_update": train_config.micro_batch_size * train_config.gradient_accumulation_steps * train_config.sequence_length,
        "max_steps": args.max_steps,
        "learning_rate": train_config.learning_rate,
        "weight_decay": train_config.weight_decay,
        "warmup_ratio": train_config.warmup_ratio,
        "train_tokens": train_dataset.metadata["token_count"],
        "validation_tokens": None if validation_loader is None else validation_loader.dataset.metadata["token_count"],
        "checkpoint_dir": str(args.checkpoint_dir or train_config.checkpoint_dir),
    }, indent=2))
    if args.dry_run:
        print("Dry run passed; no optimizer step was taken.")
        return 0
    checkpoint_manager = CheckpointManager(args.checkpoint_dir or train_config.checkpoint_dir, train_config.keep_last_checkpoints)
    engine = TrainingEngine(model, train_config, train_loader, validation_loader, device=args.device, max_steps=args.max_steps, precision=precision, checkpoint_manager=checkpoint_manager, logger=TrainingLogger(args.log_dir or train_config.log_dir), train_sampler=train_sampler)
    if args.init_from_checkpoint:
        provenance = engine.initialize_from_checkpoint(args.init_from_checkpoint)
        print(json.dumps({"initialization": provenance}, indent=2))
    elif args.resume:
        engine.load_checkpoint(None if args.resume == "latest" else Path(args.resume))
    result = engine.train()
    engine.save_checkpoint()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
