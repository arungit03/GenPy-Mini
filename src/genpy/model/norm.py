"""RMS normalization without mean subtraction or bias."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Normalize by root mean square and apply a trainable scale."""

    def __init__(self, dimension: int, epsilon: float) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply stable float32 RMS normalization and restore the input dtype."""
        source_dtype = inputs.dtype
        values = inputs.float()
        normalized = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return (normalized * self.weight.float()).to(source_dtype)
