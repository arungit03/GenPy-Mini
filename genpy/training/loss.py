"""Training-batch loss; x/y are already shifted by the memmap dataset."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def causal_batch_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=ignore_index)
