"""Reusable finite-value and gradient audits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


def is_finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def assert_finite(value: torch.Tensor, name: str = "tensor") -> None:
    if not is_finite_tensor(value):
        raise ValueError(f"non-finite values detected in {name}")


@dataclass
class GradientAudit:
    parameters_checked: int
    parameters_with_gradients: int
    parameters_without_gradients: int
    non_finite_gradients: list[str]
    global_l2_norm: float
    minimum_parameter_norm: float
    maximum_parameter_norm: float

    def to_dict(self) -> dict:
        return asdict(self)


def gradient_audit(model: nn.Module) -> GradientAudit:
    norms: list[float] = []
    non_finite: list[str] = []
    checked = with_grad = without_grad = 0
    squared_total = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        checked += 1
        if parameter.grad is None:
            without_grad += 1
            continue
        with_grad += 1
        norm = float(parameter.grad.detach().float().norm().item())
        norms.append(norm)
        squared_total += norm * norm
        if not math.isfinite(norm) or not is_finite_tensor(parameter.grad):
            non_finite.append(name)
    return GradientAudit(
        parameters_checked=checked,
        parameters_with_gradients=with_grad,
        parameters_without_gradients=without_grad,
        non_finite_gradients=non_finite,
        global_l2_norm=math.sqrt(squared_total),
        minimum_parameter_norm=min(norms, default=0.0),
        maximum_parameter_norm=max(norms, default=0.0),
    )
