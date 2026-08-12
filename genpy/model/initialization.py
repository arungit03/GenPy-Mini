"""Initialization helpers for the GenPy architecture."""

import math

import torch
from torch import nn

from genpy.config import ModelConfig
from genpy.model.rmsnorm import RMSNorm


def initialize_model(model: nn.Module, config: ModelConfig) -> None:
    """Initialize model weights and depth-scale residual output projections."""
    initialized: set[int] = set()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.weight.fill_(1.0)
            elif isinstance(module, (nn.Embedding, nn.Linear)):
                parameter_id = id(module.weight)
                if parameter_id not in initialized:
                    module.weight.normal_(mean=0.0, std=config.initializer_range)
                    initialized.add(parameter_id)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.zero_()
        residual_std = config.initializer_range / math.sqrt(2 * config.num_layers)
        for block in getattr(model, "blocks", []):
            block.attention.o_proj.weight.normal_(mean=0.0, std=residual_std)
            block.mlp.down_proj.weight.normal_(mean=0.0, std=residual_std)
