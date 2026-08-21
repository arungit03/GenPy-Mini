"""Gradient clipping and finite checks for an optimizer update."""

from __future__ import annotations

import torch


def clip_gradients(model, max_norm: float) -> float:
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite gradient norm")
    return float(norm.detach().item())


def assert_finite_gradients(model) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient in {name}")
