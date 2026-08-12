"""Root-mean-square normalization used by GenPy blocks."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Bias-free RMS normalization over the final tensor dimension."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"RMSNorm expected final dimension {self.hidden_size}, got {x.shape[-1]}"
            )
        input_dtype = x.dtype
        compute_x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        variance = compute_x.pow(2).mean(dim=-1, keepdim=True)
        normalized = compute_x * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.to(dtype=compute_x.dtype)).to(dtype=input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
