"""Single-process GenPy training engine; no model or data preprocessing lives here."""

import time

import torch
from torch import nn

from genpy.training.checkpoint import CheckpointManager
from genpy.training.logger import TrainingLogger
from genpy.training.metrics import Metrics
from genpy.training.optimizer import create_adamw
from genpy.training.precision import PrecisionManager
from genpy.training.scheduler import WarmupCosineScheduler
from genpy.training.state import TrainingState


def packed_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[:2] != targets.shape:
        raise ValueError("packed logits and targets must have shapes [B,T,V] and [B,T]")
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1).long())


class TrainingEngine:
    def __init__(self, model: nn.Module, config, train_loader, validation_loader=None, *, device="cpu", max_steps: int, optimizer=None, scheduler=None, precision=None, checkpoint_manager=None, logger=None, train_sampler=None):
        if max_steps <= 0:
            raise ValueError("max_steps must be explicitly positive")
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.max_steps = max_steps
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.train_sampler = train_sampler
        self.optimizer = optimizer or create_adamw(self.model, config)
        self.scheduler = scheduler or WarmupCosineScheduler(self.optimizer, max_steps, config.learning_rate, config.min_learning_rate, config.warmup_ratio)
        self.precision = precision or PrecisionManager(config.precision, self.device)
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger or TrainingLogger(None)
        self.state = TrainingState()
        self.metrics = Metrics()
        self.initial_loss: float | None = None
        self.continuation_provenance = None
        self._train_iterator = None

    def _next_batch(self):
        if self._train_iterator is None:
            self._train_iterator = iter(self.train_loader)
        try:
            return next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self.train_loader)
            return next(self._train_iterator)

    def _move_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return batch["input_ids"].to(self.device), batch["targets"].to(self.device)

    def validate(self, max_batches: int | None = None) -> float:
        if self.validation_loader is None:
            raise ValueError("validation_loader is not configured")
        was_training = self.model.training
        validation_sampler = getattr(self.validation_loader, "batch_sampler", None)
        sampler_state = validation_sampler.state_dict() if hasattr(validation_sampler, "state_dict") else None
        self.model.eval()
        losses = []
        try:
            with torch.no_grad():
                for index, batch in enumerate(self.validation_loader):
                    if max_batches is not None and index >= max_batches:
                        break
                    inputs, targets = self._move_batch(batch)
                    with self.precision.autocast():
                        loss = packed_loss(self.model(inputs), targets)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite validation loss")
                    losses.append(float(loss.item()))
        finally:
            if sampler_state is not None:
                validation_sampler.load_state_dict(sampler_state)
            if was_training:
                self.model.train()
        if not losses:
            raise ValueError("validation loader produced no batches")
        return sum(losses) / len(losses)

    def _check_finite_gradients(self) -> bool:
        return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()) for parameter in self.model.parameters())

    def train(self) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        final_loss = None
        while self.state.global_step < self.max_steps:
            accumulated_loss = 0.0
            last_gradient_norm = None
            for _ in range(self.config.gradient_accumulation_steps):
                batch = self._next_batch()
                inputs, targets = self._move_batch(batch)
                with self.precision.autocast():
                    logits = self.model(inputs)
                    raw_loss = packed_loss(logits, targets)
                    scaled_loss = raw_loss / self.config.gradient_accumulation_steps
                if not torch.isfinite(raw_loss):
                    raise FloatingPointError(f"non-finite training loss at global_step={self.state.global_step}")
                self.precision.backward(scaled_loss)
                accumulated_loss += float(raw_loss.detach().item())
                self.state.micro_step += 1
                self.state.samples_seen += inputs.shape[0]
                self.state.tokens_seen += inputs.numel()
            self.precision.unscale_(self.optimizer)
            if not self._check_finite_gradients():
                if self.precision.uses_grad_scaler:
                    scaler_scale = self.precision.skip_optimizer_step(self.optimizer)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.logger.log({
                        "global_step": self.state.global_step,
                        "micro_step": self.state.micro_step,
                        "tokens_seen": self.state.tokens_seen,
                        "fp16_overflow": True,
                        "skipped_update": True,
                        "scaler_scale": scaler_scale,
                    })
                    continue
                raise FloatingPointError(f"non-finite gradient at global_step={self.state.global_step}")
            if self.config.grad_clip > 0:
                last_gradient_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip).item())
            self.precision.step(self.optimizer)
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.state.global_step += 1
            self.state.optimizer_steps += 1
            final_loss = accumulated_loss / self.config.gradient_accumulation_steps
            if self.initial_loss is None:
                self.initial_loss = final_loss
            self.metrics.update(final_loss, self.config.micro_batch_size * self.config.sequence_length * self.config.gradient_accumulation_steps)
            if self.state.global_step % self.config.log_interval == 0 or self.state.global_step == 1:
                self.logger.log(self.metrics.summary(global_step=self.state.global_step, micro_step=self.state.micro_step, learning_rate=self.scheduler.get_last_lr()[0], gradient_norm=last_gradient_norm, tokens_seen=self.state.tokens_seen))
                self.metrics.reset()
            if self.validation_loader is not None and self.state.global_step % self.config.eval_interval == 0:
                validation_loss = self.validate(self.config.eval_batches)
                self.state.best_validation_loss = validation_loss if self.state.best_validation_loss is None else min(self.state.best_validation_loss, validation_loss)
                self.logger.log({"validation_loss": validation_loss, "global_step": self.state.global_step})
            if self.checkpoint_manager is not None and self.state.global_step % self.config.save_interval == 0:
                self.save_checkpoint()
        self.state.elapsed_seconds += time.perf_counter() - started
        return {"global_step": self.state.global_step, "initial_loss": self.initial_loss, "final_loss": final_loss, "tokens_seen": self.state.tokens_seen, "learning_rate": self.scheduler.get_last_lr()[0]}

    def save_checkpoint(self):
        if self.checkpoint_manager is None:
            raise ValueError("checkpoint_manager is not configured")
        data_metadata = getattr(getattr(self.train_loader, "dataset", None), "metadata", None)
        return self.checkpoint_manager.save(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, precision=self.precision, state=self.state, sampler=self.train_sampler, model_config=getattr(self.model, "config", None), training_config=self.config, max_steps=self.max_steps, data_metadata=data_metadata, provenance=self.continuation_provenance)

    def load_checkpoint(self, path=None):
        if self.checkpoint_manager is None:
            raise ValueError("checkpoint_manager is not configured")
        payload = self.checkpoint_manager.load(path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, precision=self.precision, state=self.state, sampler=self.train_sampler, model_config=getattr(self.model, "config", None), training_config=self.config)
        self._train_iterator = None
        return payload

    def initialize_from_checkpoint(self, path):
        """Initialize a fresh phase from checkpoint model weights only."""
        if self.checkpoint_manager is None:
            raise ValueError("checkpoint_manager is not configured")
        self.continuation_provenance = self.checkpoint_manager.initialize_from_checkpoint(
            path,
            model=self.model,
            model_config=getattr(self.model, "config", None),
        )
        self._train_iterator = None
        return self.continuation_provenance
