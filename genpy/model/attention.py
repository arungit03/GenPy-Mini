"""Standard causal multi-head self-attention for GenPy."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from genpy.config import ModelConfig
from genpy.model.rope import RotaryEmbedding


class GenPyAttention(nn.Module):
    """Bias-free multi-head self-attention with RoPE and a reusable mask."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        max_seq_len: int | None = None,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if config is not None:
            hidden_size = config.hidden_size
            num_heads = config.num_heads
            head_dim = config.head_dim
            max_seq_len = config.max_seq_len
            rope_theta = config.rope_theta
            attention_dropout = config.attention_dropout
            bias = config.bias
        if hidden_size is None or num_heads is None or max_seq_len is None:
            raise ValueError("attention requires hidden_size, num_heads, and max_seq_len")
        if head_dim is None:
            if hidden_size % num_heads:
                raise ValueError("hidden_size must be divisible by num_heads")
            head_dim = hidden_size // num_heads
        if hidden_size != num_heads * head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim")
        if not 0 <= attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.rope = RotaryEmbedding(head_dim, max_seq_len, rope_theta)
        self.attention_dropout = nn.Dropout(attention_dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len), persistent=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"attention expects [batch, sequence, hidden], got {tuple(x.shape)}")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"attention expected hidden size {self.hidden_size}, got {x.shape[-1]}")
        seq_len = x.shape[1]
        if seq_len <= 0 or seq_len > self.max_seq_len:
            raise ValueError(f"sequence length must be in [1, {self.max_seq_len}]")
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        q, k = self.rope(q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        attended = torch.matmul(weights, v)
        return self.o_proj(self._merge_heads(attended))

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}"
        )
