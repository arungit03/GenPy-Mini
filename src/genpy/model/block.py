"""Pre-normalized Transformer block."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from genpy.model.attention import CausalSelfAttention
from genpy.model.config import ModelConfig
from genpy.model.mlp import SwiGLU
from genpy.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Apply pre-norm causal attention and SwiGLU residual updates."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.norm_epsilon)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.norm_epsilon)
        self.mlp = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """Preserve the residual stream without in-place mutation."""
        values = inputs + self.residual_dropout(
            self.attention(self.attention_norm(inputs), attention_mask, position_ids)
        )
        output = values + self.residual_dropout(self.mlp(self.mlp_norm(values)))
        return cast(torch.Tensor, output)
