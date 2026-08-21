"""Causal multi-head self-attention with RoPE."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        bias: bool = False,
        backend: str = "sdpa",
    ) -> None:
        super().__init__()
        if d_model != n_heads * head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        if backend not in {"sdpa", "eager"}:
            raise ValueError("attention backend must be 'sdpa' or 'eager'")
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, head_dim
        self.backend = backend
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.rope = RotaryEmbedding(head_dim, max_seq_len, rope_theta)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        return x.view(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, sequence, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, sequence, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"attention expects [B, T, {self.d_model}] input")
        q, k, v = self._split_heads(self.q_proj(x)), self._split_heads(self.k_proj(x)), self._split_heads(self.v_proj(x))
        q, k = self.rope(q, k)
        dropout_p = self.dropout if self.training else 0.0
        if self.backend == "sdpa":
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            sequence = x.shape[1]
            mask = torch.triu(torch.ones(sequence, sequence, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=dropout_p, training=self.training)
            attended = torch.matmul(weights, v)
        return self.o_proj(self._merge_heads(attended))
