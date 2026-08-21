"""Rotary position embeddings for query and key tensors."""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Cached RoPE for tensors shaped ``[batch, heads, sequence, head_dim]``."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if max_seq_len <= 0 or theta <= 0:
            raise ValueError("max_seq_len and theta must be positive")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = float(theta)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", frequencies.cos(), persistent=False)
        self.register_buffer("sin_cached", frequencies.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape != k.shape:
            raise ValueError("query and key shapes must match for RoPE")
        if q.ndim != 4 or q.shape[-1] != self.head_dim:
            raise ValueError("RoPE expects [B, H, T, head_dim] tensors")
        sequence_length = q.shape[-2]
        if sequence_length > self.max_seq_len:
            raise ValueError(f"sequence length {sequence_length} exceeds RoPE limit {self.max_seq_len}")
        cos = self.cos_cached[:sequence_length].to(device=q.device, dtype=q.dtype)[None, None, :, :]
        sin = self.sin_cached[:sequence_length].to(device=q.device, dtype=q.dtype)[None, None, :, :]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            even = x[..., 0::2]
            odd = x[..., 1::2]
            return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

        return rotate(q), rotate(k)
