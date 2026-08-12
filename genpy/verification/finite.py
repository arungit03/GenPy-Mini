"""Finite-value and gradient diagnostics."""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class FiniteTensorReport:
    name: str
    is_finite: bool
    nan_count: int
    inf_count: int
    total_count: int


def finite_tensor_report(tensor: torch.Tensor, name: str = "tensor") -> FiniteTensorReport:
    """Return explicit NaN/Inf counts for one tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.is_floating_point() or tensor.is_complex():
        finite = torch.isfinite(tensor)
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int((~finite & ~torch.isnan(tensor)).sum().item())
        is_finite = bool(finite.all().item())
    else:
        nan_count = 0
        inf_count = 0
        is_finite = True
    return FiniteTensorReport(name, is_finite, nan_count, inf_count, tensor.numel())


def parameter_finite_report(model: nn.Module) -> list[FiniteTensorReport]:
    """Report finite status for every named parameter."""
    return [finite_tensor_report(parameter, name) for name, parameter in model.named_parameters()]


def gradient_report(model: nn.Module) -> dict[str, list[str] | int]:
    """Report missing and non-finite gradients by parameter name."""
    missing: list[str] = []
    non_finite: list[str] = []
    checked = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        checked += 1
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all().item()):
            non_finite.append(name)
    return {
        "checked": checked,
        "missing": missing,
        "non_finite": non_finite,
    }


def all_finite(reports: Iterable[FiniteTensorReport]) -> bool:
    """Return whether every supplied finite-value report passed."""
    return all(report.is_finite for report in reports)
