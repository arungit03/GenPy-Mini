"""Bounded rotary positional embeddings for GenPy attention."""

from __future__ import annotations

import torch
from torch import nn


def _tensor_buffer(module: nn.Module, name: str) -> torch.Tensor:
    value = module.get_buffer(name)
    if value is None:
        raise RuntimeError(f"missing RoPE buffer: {name}")
    return value


class RotaryEmbedding(nn.Module):
    """Cache fixed RoPE frequencies up to one configured context window."""

    def __init__(self, head_dimension: int, maximum_length: int, base: float) -> None:
        super().__init__()
        if head_dimension % 2:
            raise ValueError("RoPE head dimension must be even")
        inverse = 1.0 / (
            base ** (torch.arange(0, head_dimension, 2, dtype=torch.float32) / head_dimension)
        )
        positions = torch.arange(maximum_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse)
        self.maximum_length = maximum_length
        self.register_buffer("cosine", frequencies.cos(), persistent=False)
        self.register_buffer("sine", frequencies.sin(), persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate query and key tensors shaped batch, heads, sequence, dimension."""
        if position_ids.ndim != 2 or position_ids.shape != (query.shape[0], query.shape[2]):
            raise ValueError("position_ids must have shape [batch, sequence]")
        if position_ids.dtype != torch.long:
            raise TypeError("position_ids must use torch.long")
        if position_ids.numel() and (
            int(position_ids.min()) < 0 or int(position_ids.max()) >= self.maximum_length
        ):
            raise ValueError("position ID is outside the configured context")
        cosine = _tensor_buffer(self, "cosine")[position_ids].unsqueeze(1).to(query.dtype)
        sine = _tensor_buffer(self, "sine")[position_ids].unsqueeze(1).to(query.dtype)
        return _rotate(query, cosine, sine), _rotate(key, cosine, sine)


def _rotate(values: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated = torch.stack((even * cosine - odd * sine, odd * cosine + even * sine), dim=-1)
    return rotated.flatten(-2)
