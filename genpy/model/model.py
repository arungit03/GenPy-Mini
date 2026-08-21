"""Complete native PyTorch GenPy decoder-only Transformer."""

from __future__ import annotations

import torch
from torch import nn

from genpy.config import ModelConfig

from .block import TransformerBlock
from .initialization import initialize_model
from .rmsnorm import RMSNorm


class GenPyForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig, attention_backend: str = "sdpa") -> None:
        super().__init__()
        if config.tie_word_embeddings is not True:
            raise ValueError("GenPy requires tied token embeddings and LM head")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.layers = nn.ModuleList(TransformerBlock(config, attention_backend) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        initialize_model(self, n_layers=config.n_layers)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise TypeError("input_ids must be an integer tensor")
        sequence_length = input_ids.shape[1]
        if sequence_length > self.config.max_seq_len:
            raise ValueError(f"sequence length {sequence_length} exceeds context length {self.config.max_seq_len}")
        if input_ids.numel() and (input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size):
            raise ValueError("input_ids contain a token outside the configured vocabulary")
        hidden_states = self.embedding_dropout(self.token_embedding(input_ids))
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(self.final_norm(hidden_states))


GenPyModel = GenPyForCausalLM
