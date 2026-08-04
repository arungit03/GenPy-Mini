"""Aligned next-token cross-entropy for already shifted packed labels."""

from __future__ import annotations

import torch
from torch.nn import functional as functional


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Compute loss over active aligned targets without shifting labels again."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels have incompatible shapes")
    active = labels.ne(-100)
    token_count = int(active.sum().item())
    if token_count == 0:
        raise ValueError("all labels are ignored")
    loss = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
    )
    return loss, token_count
