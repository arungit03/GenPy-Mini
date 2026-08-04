"""Deterministic random initialization for GenPy models."""

from __future__ import annotations

import torch
from torch import nn

from genpy.model.norm import RMSNorm


def initialize_module(module: nn.Module, standard_deviation: float) -> None:
    """Initialize project-created parameters without external state."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=standard_deviation)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)


def seeded_model(constructor: object, seed: int) -> nn.Module:
    """Construct under an isolated deterministic CPU random-number state."""
    if not callable(constructor):
        raise TypeError("constructor must be callable")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = constructor()
    if not isinstance(model, nn.Module):
        raise TypeError("constructor did not return a module")
    return model
