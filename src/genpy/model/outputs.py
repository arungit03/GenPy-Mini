"""Typed outputs for the GenPy causal language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CausalLMOutput:
    """Output tensors and active target count from one model call."""

    logits: torch.Tensor
    loss: torch.Tensor | None
    token_count: int
