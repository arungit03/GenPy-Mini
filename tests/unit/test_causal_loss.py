from __future__ import annotations

import pytest
import torch
from torch.nn import functional as functional

from genpy.model.losses import causal_lm_loss


def test_aligned_causal_loss_matches_manual_cross_entropy() -> None:
    logits = torch.tensor([[[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]])
    labels = torch.tensor([[0, 1, -100]])
    loss, count = causal_lm_loss(logits, labels)
    expected = functional.cross_entropy(logits[:, :2].reshape(-1, 2), labels[:, :2].reshape(-1))
    torch.testing.assert_close(loss, expected)
    assert count == 2


def test_all_ignored_labels_fail() -> None:
    with pytest.raises(ValueError, match="all labels"):
        causal_lm_loss(torch.zeros(1, 2, 4), torch.full((1, 2), -100))
