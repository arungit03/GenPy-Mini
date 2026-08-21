"""Deterministic validation loop."""

from __future__ import annotations

import torch

from .loss import causal_batch_loss


def evaluate(model, batcher, device: torch.device, precision_manager) -> float:
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets in batcher:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with precision_manager.autocast():
                losses.append(float(causal_batch_loss(model(inputs), targets).float().item()))
    if was_training:
        model.train()
    if not losses:
        raise ValueError("validation batcher produced no batches")
    return sum(losses) / len(losses)
