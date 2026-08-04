from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from genpy.model.config import load_model_config
from genpy.model.transformer import build_model


def test_forward_backward_shapes_and_validation(phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    config = load_model_config(phase4_fixture["model_config"], phase4_fixture["root"])
    model = build_model(config)
    inputs = torch.randint(0, config.vocab_size, (2, config.context_length))
    labels = torch.randint(0, config.vocab_size, inputs.shape)
    labels[:, -2:] = -100
    output = model(inputs, labels=labels)
    assert output.logits.shape == (2, config.context_length, config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.token_count == 2 * (config.context_length - 2)
    output.loss.backward()
    assert model.token_embedding.weight.grad is not None
    with pytest.raises(ValueError, match="outside the vocabulary"):
        model(torch.tensor([[-1]], dtype=torch.long))
    with pytest.raises(ValueError, match="context"):
        model(torch.zeros(1, config.context_length + 1, dtype=torch.long))


def test_deterministic_random_initialization(phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    config = load_model_config(phase4_fixture["model_config"], phase4_fixture["root"])
    first = build_model(config)
    second = build_model(config)
    for left, right in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        torch.testing.assert_close(left, right)
    different = build_model(replace(config, seed=config.seed + 1))
    assert not torch.equal(first.token_embedding.weight, different.token_embedding.weight)
    assert torch.isfinite(first.token_embedding.weight).all()
    assert all(
        torch.equal(block.attention_norm.weight, torch.ones_like(block.attention_norm.weight))
        for block in first.blocks
    )
