"""Loss calculations used only by model verification."""

import torch
from torch.nn import functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute next-token cross-entropy from unshifted model inputs and labels."""
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, sequence, vocab], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [batch, sequence], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels must have matching batch and sequence dimensions: "
            f"{tuple(logits.shape[:2])} != {tuple(labels.shape)}"
        )
    if logits.shape[1] < 2:
        raise ValueError("causal LM loss requires sequence length at least 2")
    shifted_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    shifted_labels = labels[:, 1:].contiguous().view(-1)
    if shifted_labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("labels must use an integer dtype")
    return F.cross_entropy(shifted_logits, shifted_labels.long(), ignore_index=ignore_index)
