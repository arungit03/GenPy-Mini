from __future__ import annotations

import torch

from genpy.model.config import load_model_config
from genpy.model.transformer import build_model


def test_future_tokens_cannot_change_earlier_logits(phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    config = load_model_config(phase4_fixture["model_config"], phase4_fixture["root"])
    model = build_model(config).eval()
    original = torch.arange(8).unsqueeze(0) % config.vocab_size
    changed = original.clone()
    changed[:, 5:] = torch.tensor([91, 92, 93])
    with torch.no_grad():
        first = model(original).logits
        second = model(changed).logits
    torch.testing.assert_close(first[:, :5], second[:, :5], atol=1e-6, rtol=1e-6)


def test_sequence_length_one_is_finite(phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    config = load_model_config(phase4_fixture["model_config"], phase4_fixture["root"])
    output = build_model(config)(torch.tensor([[1]], dtype=torch.long))
    assert torch.isfinite(output.logits).all()
