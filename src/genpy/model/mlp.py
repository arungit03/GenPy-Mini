"""SwiGLU feed-forward network for GenPy blocks."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as functional

from genpy.model.config import ModelConfig


class SwiGLU(nn.Module):
    """Apply down(silu(gate(x)) * up(x)) with bias-free projections."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.use_bias
        )
        self.up_projection = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.use_bias
        )
        self.down_projection = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.use_bias
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return finite shape-preserving gated activations."""
        output = self.down_projection(
            functional.silu(self.gate_projection(inputs)) * self.up_projection(inputs)
        )
        return cast(torch.Tensor, output)
