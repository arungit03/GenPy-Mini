"""Validated configuration for the single-GPU training engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class EngineTrainingConfig:
    seed: int = 42
    device: str = "auto"
    precision: str = "auto"
    sequence_length: int = 1024
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int | None = None
    max_tokens: int | None = None
    pin_memory: bool = True


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass(frozen=True)
class SchedulerConfig:
    type: str = "cosine"
    warmup_steps: int = 100
    minimum_learning_rate: float = 3e-5


@dataclass(frozen=True)
class GradientConfig:
    max_norm: float = 1.0


@dataclass(frozen=True)
class ValidationConfig:
    interval_steps: int = 100
    batches: int = 20


@dataclass(frozen=True)
class LoggingConfig:
    interval_steps: int = 10


@dataclass(frozen=True)
class CheckpointConfig:
    interval_steps: int = 250
    keep_last: int = 3
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_rng: bool = True


@dataclass(frozen=True)
class TrainingEngineConfig:
    training: EngineTrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    gradients: GradientConfig
    validation: ValidationConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig

    def validate(self, require_budget: bool = False) -> None:
        t = self.training
        _positive(t.sequence_length, "training.sequence_length")
        _positive(t.micro_batch_size, "training.micro_batch_size")
        _positive(t.gradient_accumulation_steps, "training.gradient_accumulation_steps")
        if t.max_steps is not None:
            _positive(t.max_steps, "training.max_steps")
        if t.max_tokens is not None:
            _positive(t.max_tokens, "training.max_tokens")
        if t.max_steps is not None and t.max_tokens is not None:
            raise ValueError("set only one of training.max_steps or training.max_tokens")
        if require_budget and t.max_steps is None and t.max_tokens is None:
            raise ValueError("refusing to start without training.max_steps or training.max_tokens")
        if t.precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("training.precision must be auto, fp32, bf16, or fp16")
        if t.device not in {"auto", "cpu", "cuda", "cuda:0"}:
            raise ValueError("training.device must be auto, cpu, cuda, or cuda:0")
        o = self.optimizer
        if o.name.lower() != "adamw" or o.learning_rate <= 0 or o.weight_decay < 0 or o.eps <= 0:
            raise ValueError("optimizer must be AdamW with positive learning rate/epsilon")
        if not 0 <= o.beta1 < 1 or not 0 <= o.beta2 < 1:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.scheduler.type.lower() != "cosine" or self.scheduler.warmup_steps < 0 or self.scheduler.minimum_learning_rate < 0:
            raise ValueError("scheduler must be cosine with valid warmup/minimum LR")
        if self.gradients.max_norm <= 0:
            raise ValueError("gradients.max_norm must be positive")


def load_training_config(path: str | Path) -> TrainingEngineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    def section(name: str) -> dict:
        value = raw.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")
        return value
    t, o, s, g, v, l, c = map(section, ("training", "optimizer", "scheduler", "gradients", "validation", "logging", "checkpoint"))
    config = TrainingEngineConfig(
        training=EngineTrainingConfig(**{key: t[key] for key in EngineTrainingConfig.__dataclass_fields__ if key in t}),
        optimizer=OptimizerConfig(**{key: o[key] for key in OptimizerConfig.__dataclass_fields__ if key in o}),
        scheduler=SchedulerConfig(**{key: s[key] for key in SchedulerConfig.__dataclass_fields__ if key in s}),
        gradients=GradientConfig(**{key: g[key] for key in GradientConfig.__dataclass_fields__ if key in g}),
        validation=ValidationConfig(**{key: v[key] for key in ValidationConfig.__dataclass_fields__ if key in v}),
        logging=LoggingConfig(**{key: l[key] for key in LoggingConfig.__dataclass_fields__ if key in l}),
        checkpoint=CheckpointConfig(**{key: c[key] for key in CheckpointConfig.__dataclass_fields__ if key in c}),
    )
    config.validate()
    return config
