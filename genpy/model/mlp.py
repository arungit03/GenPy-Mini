"""SwiGLU feed-forward network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_hidden_size: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_hidden_size, bias=bias)
        self.up_proj = nn.Linear(d_model, ffn_hidden_size, bias=bias)
        self.down_proj = nn.Linear(ffn_hidden_size, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
