"""Warmup plus cosine learning-rate scheduler stepped per optimizer update."""

from __future__ import annotations

import math


class WarmupCosineScheduler:
    def __init__(self, optimizer, max_lr: float, minimum_lr: float, warmup_steps: int, total_steps: int) -> None:
        if total_steps <= 0 or warmup_steps < 0 or minimum_lr < 0:
            raise ValueError("invalid scheduler configuration")
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.minimum_lr = minimum_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_num = 0
        self._set_lr(self.lr_at(0))

    def lr_at(self, step: int) -> float:
        if self.warmup_steps and step <= self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        if step >= self.total_steps:
            return self.minimum_lr
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.minimum_lr + 0.5 * (self.max_lr - self.minimum_lr) * (1.0 + math.cos(math.pi * progress))

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def step(self) -> None:
        self.step_num += 1
        self._set_lr(self.lr_at(self.step_num))

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"step_num": self.step_num, "max_lr": self.max_lr, "minimum_lr": self.minimum_lr, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}

    def load_state_dict(self, state: dict) -> None:
        self.step_num = int(state["step_num"])
        self.max_lr = float(state.get("max_lr", self.max_lr))
        self.minimum_lr = float(state.get("minimum_lr", self.minimum_lr))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.total_steps = int(state.get("total_steps", self.total_steps))
        self._set_lr(self.lr_at(self.step_num))
