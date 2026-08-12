"""SwiGLU feed-forward network."""

import torch
from torch import nn
from torch.nn import functional as F

from genpy.config import ModelConfig


class SwiGLU(nn.Module):
    """Bias-free SwiGLU: down(SiLU(gate(x)) * up(x))."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if config is not None:
            hidden_size = config.hidden_size
            intermediate_size = config.intermediate_size
            bias = config.bias
        if hidden_size is None or intermediate_size is None:
            raise ValueError("SwiGLU requires hidden_size and intermediate_size")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"
