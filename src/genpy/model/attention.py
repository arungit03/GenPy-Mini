"""Bias-free multi-head causal self-attention."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as functional

from genpy.model.config import ModelConfig
from genpy.model.rotary import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """Standard causal MHA with RoPE and optional padding masks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dimension = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dimension = config.head_dimension
        self.dropout = config.attention_dropout
        self.query = nn.Linear(dimension, dimension, bias=config.use_bias)
        self.key = nn.Linear(dimension, dimension, bias=config.use_bias)
        self.value = nn.Linear(dimension, dimension, bias=config.use_bias)
        self.output = nn.Linear(dimension, dimension, bias=config.use_bias)
        self.rotary = RotaryEmbedding(
            config.head_dimension, config.context_length, config.rope_base
        )

    def _project(self, inputs: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        batch, sequence, _ = inputs.shape
        projected = cast(torch.Tensor, projection(inputs))
        return projected.view(
            batch, sequence, self.num_heads, self.head_dimension
        ).transpose(1, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        use_reference: bool = False,
    ) -> torch.Tensor:
        """Apply causal attention without a persistent square mask."""
        query = self._project(inputs, self.query)
        key = self._project(inputs, self.key)
        value = self._project(inputs, self.value)
        query, key = self.rotary(query, key, position_ids)
        sequence = inputs.shape[1]
        causal = torch.ones((sequence, sequence), device=inputs.device, dtype=torch.bool).tril()
        allowed = causal.view(1, 1, sequence, sequence) & attention_mask[:, None, None, :]
        dropout = self.dropout if self.training else 0.0
        if use_reference:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
                self.head_dimension
            )
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            probabilities = functional.softmax(scores, dim=-1)
            probabilities = functional.dropout(probabilities, dropout, self.training)
            attended = torch.matmul(probabilities, value)
        else:
            attended = functional.scaled_dot_product_attention(
                query, key, value, attn_mask=allowed, dropout_p=dropout
            )
        merged = attended.transpose(1, 2).contiguous().view(inputs.shape)
        return cast(torch.Tensor, self.output(merged))
