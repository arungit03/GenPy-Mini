from __future__ import annotations

import torch

from genpy.model.rotary import RotaryEmbedding


def test_rope_shapes_determinism_and_position_sensitivity() -> None:
    rope = RotaryEmbedding(8, 16, 10000.0)
    query = torch.arange(64, dtype=torch.float32).view(1, 2, 4, 8)
    positions = torch.arange(4).unsqueeze(0)
    first, key = rope(query, query, positions)
    second, _ = rope(query, query, positions)
    assert first.shape == query.shape
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, key)
    assert not torch.equal(first[:, :, 0], first[:, :, 1])
    assert "cosine" not in rope.state_dict()


def test_rope_rejects_wrapped_positions() -> None:
    rope = RotaryEmbedding(8, 4, 10000.0)
    values = torch.zeros(1, 1, 1, 8)
    try:
        rope(values, values, torch.tensor([[4]]))
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("out-of-range position was accepted")
