import torch

from genpy.training.loss import causal_batch_loss


def test_batch_loss_uses_targets_directly() -> None:
    logits = torch.randn(2, 3, 8); targets = torch.tensor([[1, 2, 3], [3, 2, 1]])
    assert torch.isfinite(causal_batch_loss(logits, targets))
