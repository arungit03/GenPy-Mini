"""Minimal, correctly shifted causal language-model loss."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [B,T,V] and input_ids must be [B,T]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids batch/sequence dimensions must match")
    if input_ids.shape[1] < 2:
        raise ValueError("causal loss requires sequence length >= 2")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, logits.shape[-1]), shift_labels.view(-1), ignore_index=ignore_index)


def reference_causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    losses = F.cross_entropy(shift_logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1), ignore_index=ignore_index)
    return losses
