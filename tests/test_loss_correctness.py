import pytest
import torch
from torch.nn import functional as F

from genpy.verification import causal_lm_loss


def test_causal_lm_loss_matches_manual_reference():
    torch.manual_seed(11)
    logits = torch.randn(2, 5, 7)
    labels = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    actual = causal_lm_loss(logits, labels)
    shifted_logits = logits[:, :-1].reshape(-1, 7)
    shifted_labels = labels[:, 1:].reshape(-1)
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    reference = -log_probs.gather(1, shifted_labels[:, None]).mean()
    assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-6)


def test_causal_lm_loss_uses_next_token_shift():
    logits = torch.full((1, 3, 4), -10.0)
    labels = torch.tensor([[0, 1, 2]])
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    assert causal_lm_loss(logits, labels) < 1e-5
    wrong_unshifted = F.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1))
    assert causal_lm_loss(logits, labels) < wrong_unshifted


def test_causal_lm_loss_rejects_incompatible_shapes():
    with pytest.raises(ValueError):
        causal_lm_loss(torch.randn(2, 4, 8), torch.zeros(2, 3, dtype=torch.long))
    with pytest.raises(ValueError):
        causal_lm_loss(torch.randn(2, 1, 8), torch.zeros(2, 1, dtype=torch.long))
