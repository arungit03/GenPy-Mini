"""Rotary position embeddings for attention queries and keys."""

import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs: (x0, x1) -> (-x1, x0)."""
    even = x[..., ::2]
    odd = x[..., 1::2]
    rotated = torch.stack((-odd, even), dim=-1)
    return rotated.flatten(start_dim=-2)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a cosine/sine rotation to Q and K, leaving V untouched by design.

    Q and K are expected to end in ``head_dim`` and use either ``[B,H,T,D]``
    or another layout whose sequence dimension is represented by the cache.
    The cache is broadcast across all leading dimensions.
    """
    if q.shape != k.shape:
        raise ValueError("q and k must have identical shapes")
    if q.shape[-1] % 2:
        raise ValueError("RoPE requires an even head dimension")
    if sin.shape != cos.shape:
        raise ValueError("RoPE cache has incompatible shape")
    if cos.shape[-1] not in (q.shape[-1] // 2, q.shape[-1]):
        raise ValueError("RoPE cache has incompatible head dimension")
    if cos.shape[-2] != q.shape[-2]:
        raise ValueError("RoPE cache sequence length must match q and k")
    cos_full = cos if cos.shape[-1] == q.shape[-1] else torch.repeat_interleave(cos, 2, dim=-1)
    sin_full = sin if sin.shape[-1] == q.shape[-1] else torch.repeat_interleave(sin, 2, dim=-1)
    cos_full = cos_full.to(device=q.device, dtype=q.dtype)
    sin_full = sin_full.to(device=q.device, dtype=q.dtype)
    while cos_full.ndim < q.ndim:
        cos_full = cos_full.unsqueeze(0)
        sin_full = sin_full.unsqueeze(0)
    q_out = q * cos_full + _rotate_half(q) * sin_full
    k_out = k * cos_full + _rotate_half(k) * sin_full
    return q_out, k_out


class RotaryEmbedding(nn.Module):
    """Non-trainable RoPE frequency and cosine/sine cache."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("RoPE head_dim must be a positive even integer")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if theta <= 0:
            raise ValueError("RoPE theta must be positive")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = float(theta)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        angles = torch.einsum("t,d->td", positions, inverse_frequency)
        self.register_buffer("inv_freq", inverse_frequency, persistent=False)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def get_cos_sin(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= 0:
            raise ValueError("sequence length must be positive")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds RoPE maximum {self.max_seq_len}"
            )
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape != k.shape:
            raise ValueError("q and k must have identical shapes")
        if q.shape[-1] != self.head_dim:
            raise ValueError("q and k head dimension does not match RoPE")
        cos, sin = self.get_cos_sin(q.shape[-2])
        return apply_rotary_pos_emb(q, k, cos, sin)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, theta={self.theta}"
