"""Random initialization for the GenPy Transformer."""

from __future__ import annotations

import math

import torch
from torch import nn

from .rmsnorm import RMSNorm


def initialize_model(model: nn.Module, base_std: float = 0.02, n_layers: int | None = None) -> None:
    """Initialize weights in-place; residual projections use scaled normal noise."""
    scaled_std = base_std / math.sqrt(2 * n_layers) if n_layers else base_std
    initialized: set[int] = set()
    with torch.no_grad():
        for module_name, module in model.named_modules():
            if isinstance(module, RMSNorm):
                module.weight.fill_(1.0)
                continue
            if isinstance(module, (nn.Embedding, nn.Linear)) and module.weight is not None:
                if id(module.weight) in initialized:
                    continue
                std = scaled_std if module_name.endswith("o_proj") or module_name.endswith("down_proj") else base_std
                nn.init.normal_(module.weight, mean=0.0, std=std)
                initialized.add(id(module.weight))
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
