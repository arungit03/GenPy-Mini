"""Configuration-driven GenPy decoder-only Transformer."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from genpy.model.block import TransformerBlock
from genpy.model.config import ModelConfig
from genpy.model.initialization import initialize_module, seeded_model
from genpy.model.losses import causal_lm_loss
from genpy.model.norm import RMSNorm
from genpy.model.outputs import CausalLMOutput


class GenPyForCausalLM(nn.Module):
    """Decoder-only causal language model with tied token and output weights."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(lambda module: initialize_module(module, config.initializer_std))
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> None:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise TypeError("input_ids must be torch.long with shape [batch, sequence]")
        if input_ids.shape[1] == 0 or input_ids.shape[1] > self.config.context_length:
            raise ValueError("input sequence length is outside the configured context")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError("input token ID is outside the vocabulary")
        if labels is not None:
            if labels.shape != input_ids.shape or labels.dtype != torch.long:
                raise TypeError("labels must be torch.long and match input_ids")
            valid = labels.eq(-100) | ((labels >= 0) & (labels < self.config.vocab_size))
            if not bool(valid.all()):
                raise ValueError("label token ID is outside the vocabulary")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if position_ids is not None and position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must match input_ids")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        """Return logits and optional aligned next-token loss without truncation."""
        self._validate_inputs(input_ids, labels, attention_mask, position_ids)
        batch, sequence = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)
        if position_ids is None:
            position_ids = torch.arange(sequence, device=input_ids.device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        elif position_ids.dtype != torch.long:
            raise TypeError("position_ids must use torch.long")
        if position_ids.numel() and (
            int(position_ids.min()) < 0 or int(position_ids.max()) >= self.config.context_length
        ):
            raise ValueError("position ID is outside the configured context")
        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                function: Callable[..., torch.Tensor] = block
                hidden = checkpoint(
                    function,
                    hidden,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                hidden = block(hidden, attention_mask, position_ids)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        token_count = 0
        if labels is not None:
            loss, token_count = causal_lm_loss(logits, labels)
        return CausalLMOutput(logits=logits, loss=loss, token_count=token_count)


def build_model(config: ModelConfig, *, device: str | torch.device = "cpu") -> GenPyForCausalLM:
    """Build a deterministically initialized model on the requested local device."""
    model = seeded_model(lambda: GenPyForCausalLM(config), config.seed)
    assert isinstance(model, GenPyForCausalLM)
    return model.to(device)
