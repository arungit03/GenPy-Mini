"""Resumable single-device training engine; no production budget is implied."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from genpy.model.utils import count_parameters

from .batcher import RandomWindowBatcher, SequentialValidationBatcher
from .checkpoint import CheckpointManager
from .config import TrainingEngineConfig
from .dataset import MemmapTokenDataset
from .gradients import assert_finite_gradients, clip_gradients
from .logger import MetricsLogger
from .loss import causal_batch_loss
from .optimizer import create_adamw
from .precision import PrecisionManager
from .scheduler import WarmupCosineScheduler
from .state import TrainingState
from .validation import evaluate


class TrainingEngine:
    def __init__(self, model, train_dataset: MemmapTokenDataset, validation_dataset: MemmapTokenDataset, config: TrainingEngineConfig, run_dir: str | Path, model_config_hash: str = "unknown", cache_hash: str = "unknown", tokenizer_hash: str = "unknown", run_id: str = "genpy200m_pretrain_v1") -> None:
        config.validate()
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.training.device)
        self.model.to(self.device)
        self.train_dataset, self.validation_dataset = train_dataset, validation_dataset
        self.precision = PrecisionManager(config.training.precision, self.device)
        self.optimizer, self.optimizer_audit = create_adamw(model, config.optimizer)
        total_steps = config.training.max_steps or max(1, (config.training.max_tokens or config.training.sequence_length) // (config.training.sequence_length * config.training.micro_batch_size * config.training.gradient_accumulation_steps))
        self.scheduler = WarmupCosineScheduler(self.optimizer, config.optimizer.learning_rate, config.scheduler.minimum_learning_rate, config.scheduler.warmup_steps, total_steps)
        self.train_batcher = RandomWindowBatcher(train_dataset, config.training.micro_batch_size, config.training.seed)
        self.validation_batcher = SequentialValidationBatcher(validation_dataset, config.training.micro_batch_size, config.validation.batches)
        self.state = TrainingState(run_id=run_id, current_learning_rate=self.scheduler.get_last_lr()[0])
        self.checkpoints = CheckpointManager(Path(run_dir) / "checkpoints", config.checkpoint.keep_last)
        self.logger = MetricsLogger(Path(run_dir) / "logs/training_metrics.jsonl")
        self.metadata = {"model_config_hash": model_config_hash, "cache_hash": cache_hash, "tokenizer_hash": tokenizer_hash, "parameter_count": count_parameters(model), "sequence_length": config.training.sequence_length, "precision": self.precision.mode}

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cpu": return torch.device("cpu")
        if requested in {"cuda", "cuda:0"}:
            if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
            return torch.device("cuda:0")
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def effective_tokens_per_update(self) -> int:
        t = self.config.training
        return t.sequence_length * t.micro_batch_size * t.gradient_accumulation_steps

    def train_optimizer_step(self) -> dict:
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        losses = []
        start = time.perf_counter()
        for _ in range(self.config.training.gradient_accumulation_steps):
            inputs, targets = self.train_batcher.next_batch()
            self.state.windows_seen += inputs.shape[0]
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            with self.precision.autocast():
                loss = causal_batch_loss(self.model(inputs), targets)
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {self.state.global_step + 1}")
            losses.append(float(loss.detach().item()))
            self.precision.backward(loss / self.config.training.gradient_accumulation_steps)
        self.precision.unscale(self.optimizer)
        assert_finite_gradients(self.model)
        gradient_norm = clip_gradients(self.model, self.config.gradients.max_norm)
        self.precision.step(self.optimizer); self.scheduler.step()
        self.state.global_step += 1
        self.state.tokens_seen += self.effective_tokens_per_update
        self.state.current_learning_rate = self.scheduler.get_last_lr()[0]
        result = {"step": self.state.global_step, "tokens_seen": self.state.tokens_seen, "train_loss": sum(losses) / len(losses), "learning_rate": self.state.current_learning_rate, "gradient_norm": gradient_norm, "precision": self.precision.mode, "tokens_per_second": self.effective_tokens_per_update / max(1e-9, time.perf_counter() - start)}
        self.logger.log(result)
        return result

    def validate(self) -> float:
        value = evaluate(self.model, self.validation_batcher, self.device, self.precision)
        self.state.last_validation_loss = value
        if self.state.best_validation_loss is None or value < self.state.best_validation_loss: self.state.best_validation_loss = value
        self.logger.log({"step": self.state.global_step, "validation_loss": value, "best_validation_loss": self.state.best_validation_loss})
        return value

    def save_checkpoint(self) -> Path:
        return self.checkpoints.save(self.model, self.optimizer, self.scheduler, self.precision, self.state, self.train_batcher, self.metadata)

    def resume(self, checkpoint: str | Path = "auto") -> TrainingState:
        path = self.checkpoints.latest() if checkpoint == "auto" else Path(checkpoint)
        if path is None: raise FileNotFoundError("no valid checkpoint available for resume")
        self.state = self.checkpoints.load(path, self.model, self.optimizer, self.scheduler, self.precision, self.state, self.train_batcher, self.metadata)
        self.model.to(self.device)
        self.logger.log({"event": "resume", "step": self.state.global_step, "tokens_seen": self.state.tokens_seen})
        return self.state

    def run(self) -> TrainingState:
        self.config.validate(require_budget=True)
        target_steps = self.config.training.max_steps
        while target_steps is None or self.state.global_step < target_steps:
            self.train_optimizer_step()
            if self.state.global_step % self.config.validation.interval_steps == 0: self.validate()
            if self.state.global_step % self.config.checkpoint.interval_steps == 0: self.save_checkpoint()
            if self.config.training.max_tokens is not None and self.state.tokens_seen >= self.config.training.max_tokens: break
        self.validate(); self.save_checkpoint()
        return self.state
