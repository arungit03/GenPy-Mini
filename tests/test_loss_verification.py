import torch

from genpy.verification.loss import causal_lm_loss, reference_causal_lm_loss


def test_causal_loss_matches_reference_and_supports_ignore_index() -> None:
    torch.manual_seed(2)
    logits = torch.randn(2, 5, 17)
    labels = torch.randint(0, 17, (2, 5))
    assert torch.allclose(causal_lm_loss(logits, labels), reference_causal_lm_loss(logits, labels), atol=1e-7)
    ignored = labels.clone(); ignored[:, 2] = -100
    assert torch.isfinite(causal_lm_loss(logits, ignored, ignore_index=-100))
