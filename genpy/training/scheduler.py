"""Optimizer-step linear warmup and cosine learning-rate schedule."""

import math

import torch


def warmup_cosine_lr(step: int, total_steps: int, learning_rate: float, min_learning_rate: float, warmup_steps: int) -> float:
    if total_steps <= 0 or step < 0:
        raise ValueError("total_steps must be positive and step non-negative")
    if learning_rate <= 0 or min_learning_rate < 0 or min_learning_rate > learning_rate:
        raise ValueError("learning-rate bounds are invalid")
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    if step >= total_steps - 1:
        return min_learning_rate
    decay_steps = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + (learning_rate - min_learning_rate) * cosine


class WarmupCosineScheduler:
    """Small serializable scheduler stepped once per optimizer update."""

    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int, learning_rate: float, min_learning_rate: float, warmup_ratio: float) -> None:
        if not 0 <= warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.step_count = 0
        self._set_lr(warmup_cosine_lr(0, total_steps, learning_rate, min_learning_rate, self.warmup_steps))

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def step(self) -> None:
        self.step_count += 1
        value = warmup_cosine_lr(min(self.step_count, self.total_steps - 1), self.total_steps, self.learning_rate, self.min_learning_rate, self.warmup_steps)
        self._set_lr(value)

    def state_dict(self) -> dict:
        return {"total_steps": self.total_steps, "learning_rate": self.learning_rate, "min_learning_rate": self.min_learning_rate, "warmup_steps": self.warmup_steps, "step_count": self.step_count}

    def load_state_dict(self, state: dict) -> None:
        expected = (self.total_steps, self.learning_rate, self.min_learning_rate, self.warmup_steps)
        actual = (state["total_steps"], state["learning_rate"], state["min_learning_rate"], state["warmup_steps"])
        if expected != actual:
            raise ValueError("scheduler configuration mismatch")
        self.step_count = int(state["step_count"])
        current_index = min(self.step_count, self.total_steps - 1)
        self._set_lr(warmup_cosine_lr(current_index, self.total_steps, self.learning_rate, self.min_learning_rate, self.warmup_steps))
