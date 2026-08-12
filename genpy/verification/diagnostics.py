"""Compact model diagnostics for verification scripts and reports."""

from typing import Any

import torch
from torch import nn

from genpy.verification.finite import gradient_report, parameter_finite_report


def model_diagnostics(model: nn.Module) -> dict[str, Any]:
    """Collect architecture-independent diagnostics without training behavior."""
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    tying = False
    if hasattr(model, "lm_head") and hasattr(model, "token_embedding"):
        tying = model.lm_head.weight is model.token_embedding.weight
    return {
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "parameters_finite": all(report.is_finite for report in parameter_finite_report(model)),
        "gradient_report": gradient_report(model),
        "weight_tied": tying,
        "config": getattr(model, "config", None),
    }


def bias_free_linear_names(model: nn.Module) -> list[str]:
    """Return names of linear modules that unexpectedly contain a bias."""
    return [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.bias is not None
    ]


def finite_tensor(tensor: torch.Tensor) -> bool:
    """Small convenience predicate for script assertions."""
    return bool(torch.isfinite(tensor).all().item())
