"""Pre-norm GenPy Transformer decoder block."""

import torch
from torch import nn

from genpy.config import ModelConfig
from genpy.model.attention import GenPyAttention
from genpy.model.rmsnorm import RMSNorm
from genpy.model.swiglu import SwiGLU


class GenPyBlock(nn.Module):
    """RMSNorm → causal attention → residual → RMSNorm → SwiGLU → residual."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        intermediate_size: int | None = None,
        max_seq_len: int | None = None,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if config is not None:
            hidden_size = config.hidden_size
            num_heads = config.num_heads
            head_dim = config.head_dim
            intermediate_size = config.intermediate_size
            max_seq_len = config.max_seq_len
            norm_eps = config.norm_eps
            rope_theta = config.rope_theta
            attention_dropout = config.attention_dropout
            residual_dropout = config.residual_dropout
            bias = config.bias
        if hidden_size is None or num_heads is None or head_dim is None or intermediate_size is None or max_seq_len is None:
            raise ValueError("block requires complete model dimensions")
        self.hidden_size = hidden_size
        self.attn_norm = RMSNorm(hidden_size, norm_eps)
        self.attention = GenPyAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            bias=bias,
        )
        self.ffn_norm = RMSNorm(hidden_size, norm_eps)
        self.mlp = SwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size, bias=bias)
        if not 0 <= residual_dropout < 1:
            raise ValueError("residual_dropout must be in [0, 1)")
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.attention(self.attn_norm(x)))
        x = x + self.residual_dropout(self.mlp(self.ffn_norm(x)))
        return x

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}"
