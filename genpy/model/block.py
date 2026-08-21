"""Pre-normalized GenPy decoder block."""

from __future__ import annotations

import torch
from torch import nn

from genpy.config import ModelConfig

from .attention import CausalSelfAttention
from .mlp import SwiGLU
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, attention_backend: str = "sdpa") -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(
            config.d_model, config.n_heads, config.head_dim, config.max_seq_len,
            config.rope_theta, config.attention_dropout, config.attention_bias, attention_backend,
        )
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config.d_model, config.ffn_hidden_size, config.mlp_bias)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.attention(self.attn_norm(x)))
        x = x + self.residual_dropout(self.mlp(self.ffn_norm(x)))
        return x
