from __future__ import annotations

import torch
from torch.nn import functional as functional

from genpy.model.config import load_model_config
from genpy.model.mlp import SwiGLU


def test_swiglu_matches_documented_formula_and_has_gradients(phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    config = load_model_config(phase4_fixture["model_config"], phase4_fixture["root"])
    mlp = SwiGLU(config)
    inputs = torch.randn(2, 3, config.hidden_size, requires_grad=True)
    actual = mlp(inputs)
    expected = mlp.down_projection(
        functional.silu(mlp.gate_projection(inputs)) * mlp.up_projection(inputs)
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
