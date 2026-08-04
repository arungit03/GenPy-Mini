from __future__ import annotations

import torch

from genpy.model.attention import CausalSelfAttention
from genpy.model.config import load_model_config


def test_optimized_attention_matches_reference(phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    config = load_model_config(phase4_fixture["model_config"], phase4_fixture["root"])
    attention = CausalSelfAttention(config).eval()
    inputs = torch.randn(2, 8, config.hidden_size)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[1, -2:] = False
    positions = torch.arange(8).repeat(2, 1)
    optimized = attention(inputs, mask, positions)
    reference = attention(inputs, mask, positions, use_reference=True)
    torch.testing.assert_close(optimized, reference, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(optimized).all()
