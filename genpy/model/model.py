"""Full GenPy decoder-only language model."""

from collections.abc import Mapping

import torch
from torch import nn

from genpy.config import ModelConfig
from genpy.model.block import GenPyBlock
from genpy.model.initialization import initialize_model
from genpy.model.rmsnorm import RMSNorm


class GenPyForCausalLM(nn.Module):
    """GenPy decoder with a tied token embedding and language-model head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.blocks = nn.ModuleList([GenPyBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.bias)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        initialize_model(self, config)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "GenPyForCausalLM":
        return cls(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have rank 2 [batch, sequence], got {input_ids.ndim}")
        if input_ids.shape[1] <= 0 or input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"input sequence length must be in [1, {self.config.max_seq_len}]"
            )
        if input_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("input_ids must use an integer dtype")
        if input_ids.numel():
            minimum = int(input_ids.min().item())
            maximum = int(input_ids.max().item())
            if minimum < 0 or maximum >= self.config.vocab_size:
                raise ValueError(
                    f"input_ids must be in [0, {self.config.vocab_size - 1}], "
                    f"got range [{minimum}, {maximum}]"
                )
        x = self.embedding_dropout(self.token_embedding(input_ids.long()))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def parameter_breakdown(self) -> Mapping[str, int]:
        """Count unique trainable parameters by major architectural component."""
        groups = {"Embedding": 0, "Attention": 0, "SwiGLU": 0, "RMSNorm": 0, "Other": 0}
        seen: set[int] = set()
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if "token_embedding" in name:
                group = "Embedding"
            elif ".attention." in name:
                group = "Attention"
            elif ".mlp." in name:
                group = "SwiGLU"
            elif "norm" in name:
                group = "RMSNorm"
            else:
                group = "Other"
            groups[group] += parameter.numel()
        return groups

    def extra_repr(self) -> str:
        return (
            f"name={self.config.name}, vocab_size={self.config.vocab_size}, "
            f"max_seq_len={self.config.max_seq_len}, hidden_size={self.config.hidden_size}, "
            f"layers={self.config.num_layers}, heads={self.config.num_heads}, "
            f"head_dim={self.config.head_dim}, intermediate_size={self.config.intermediate_size}"
        )
