"""SFT trainer reusing the production precision, optimizer, and checkpoint code."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from genpy.model.utils import count_parameters

from .checkpoint import CheckpointManager
from .config import TrainingEngineConfig
from .gradients import assert_finite_gradients, clip_gradients
from .logger import MetricsLogger
from .loss import causal_batch_loss
from .optimizer import create_adamw
from .precision import PrecisionManager
from .scheduler import WarmupCosineScheduler
from .sft_dataset import SFTMemmapDataset, SFTRandomBatcher, SFTSequentialBatcher
from .state import TrainingState


class SFTTrainingEngine:
    def __init__(self, model, train_dataset: SFTMemmapDataset, validation_dataset: SFTMemmapDataset, config: TrainingEngineConfig, run_dir: str | Path, base_model_hash: str, sft_manifest_hash: str, run_id: str = "genpy200m_sft_v1") -> None:
        config.validate(require_budget=True)
        if config.training.max_steps is None:
            raise ValueError("SFT config requires a fixed max_steps after token-cache analysis")
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.training.device)
        self.model.to(self.device)
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.precision = PrecisionManager(config.training.precision, self.device)
        self.optimizer, self.optimizer_audit = create_adamw(model, config.optimizer)
        self.scheduler = WarmupCosineScheduler(self.optimizer, config.optimizer.learning_rate, config.scheduler.minimum_learning_rate, config.scheduler.warmup_steps, config.training.max_steps)
        self.train_batcher = SFTRandomBatcher(train_dataset, config.training.micro_batch_size, config.training.seed)
        self.validation_batcher = SFTSequentialBatcher(validation_dataset, config.training.micro_batch_size, config.validation.batches)
        self.state = TrainingState(run_id=run_id, current_learning_rate=self.scheduler.get_last_lr()[0])
        self.run_dir = Path(run_dir)
        self.checkpoints = CheckpointManager(self.run_dir / "checkpoints", config.checkpoint.keep_last)
        self.logger = MetricsLogger(self.run_dir / "logs/training_metrics.jsonl")
        self.metadata = {"base_model_hash": base_model_hash, "sft_manifest_hash": sft_manifest_hash, "parameter_count": count_parameters(model), "sequence_length": config.training.sequence_length, "precision": self.precision.mode, "scheduler_total_steps": config.training.max_steps}
        self.last_run_status = "NOT_STARTED"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.run_dir / "run_manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps({"run_id": run_id, "metadata": self.metadata, "training_config": asdict(config)}, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cpu":
            return torch.device("cpu")
        if requested in {"cuda", "cuda:0"}:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            return torch.device("cuda:0")
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def effective_tokens_per_update(self) -> int:
        t = self.config.training
        return t.sequence_length * t.micro_batch_size * t.gradient_accumulation_steps

    def train_optimizer_step(self) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses, supervised_tokens = [], 0
        start = time.perf_counter()
        for _ in range(self.config.training.gradient_accumulation_steps):
            inputs, targets = self.train_batcher.next_batch()
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            supervised_tokens += int(targets.ne(-100).sum().item())
            with self.precision.autocast():
                loss = causal_batch_loss(self.model(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite SFT loss at step {self.state.global_step + 1}")
            losses.append(float(loss.detach().item()))
            self.precision.backward(loss / self.config.training.gradient_accumulation_steps)
        self.precision.unscale(self.optimizer)
        assert_finite_gradients(self.model)
        gradient_norm = clip_gradients(self.model, self.config.gradients.max_norm)
        self.precision.step(self.optimizer)
        self.scheduler.step()
        self.state.global_step += 1
        self.state.tokens_seen += supervised_tokens
        self.state.current_learning_rate = self.scheduler.get_last_lr()[0]
        elapsed = time.perf_counter() - start
        result = {"step": self.state.global_step, "global_step": self.state.global_step, "tokens_seen": self.state.tokens_seen, "supervised_tokens": supervised_tokens, "train_loss": sum(losses) / len(losses), "learning_rate": self.state.current_learning_rate, "gradient_norm": gradient_norm, "tokens_per_second": self.effective_tokens_per_update / max(elapsed, 1e-9), "step_time_seconds": elapsed, "precision": self.precision.mode}
        self.logger.log(result)
        return result

    def validate(self) -> float:
        was_training = self.model.training
        self.model.eval()
        values = []
        with torch.no_grad():
            for inputs, targets in self.validation_batcher.batches_iter():
                with self.precision.autocast():
                    values.append(float(causal_batch_loss(self.model(inputs.to(self.device)), targets.to(self.device)).item()))
        if was_training:
            self.model.train()
        if not values:
            raise ValueError("validation dataset produced no batches")
        value = sum(values) / len(values)
        self.state.last_validation_loss = value
        if self.state.best_validation_loss is None or value < self.state.best_validation_loss:
            self.state.best_validation_loss = value
        self.logger.log({"step": self.state.global_step, "validation_loss": value, "best_validation_loss": self.state.best_validation_loss, "learning_rate": self.state.current_learning_rate, "precision": self.precision.mode})
        return value

    def save_checkpoint(self) -> Path:
        return self.checkpoints.save(self.model, self.optimizer, self.scheduler, self.precision, self.state, self.train_batcher, self.metadata)

    def resume(self, checkpoint: str | Path = "auto") -> TrainingState:
        path = self.checkpoints.latest() if checkpoint == "auto" else Path(checkpoint)
        if path is None:
            raise FileNotFoundError("no valid SFT checkpoint available for resume")
        self.state = self.checkpoints.load(path, self.model, self.optimizer, self.scheduler, self.precision, self.state, self.train_batcher, self.metadata)
        self.model.to(self.device)
        self.logger.log({"event": "resume", "step": self.state.global_step, "tokens_seen": self.state.tokens_seen})
        return self.state

    def run(self, session_steps: int | None = None) -> TrainingState:
        if session_steps is not None and (isinstance(session_steps, bool) or session_steps <= 0):
            raise ValueError("session_steps must be positive")
        start = self.state.global_step
        while self.state.global_step < self.config.training.max_steps and (session_steps is None or self.state.global_step - start < session_steps):
            self.train_optimizer_step()
            if self.state.global_step % self.config.validation.interval_steps == 0:
                self.validate()
            if self.state.global_step % self.config.checkpoint.interval_steps == 0:
                self.save_checkpoint()
        if self.state.global_step == start:
            raise RuntimeError("SFT training made no optimizer progress")
        self.validate()
        self.save_checkpoint()
        self.last_run_status = "TRAINING_COMPLETE" if self.state.global_step >= self.config.training.max_steps else "SESSION_COMPLETE"
        return self.state
