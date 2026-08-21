import pytest
import torch

from genpy.model import RotaryEmbedding


def test_rope_shape_rotation_and_boundary() -> None:
    rope = RotaryEmbedding(8, max_seq_len=4)
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn_like(q)
    rotated_q, rotated_k = rope(q, k)
    assert rotated_q.shape == q.shape and rotated_k.shape == k.shape
    assert torch.allclose(rotated_q[:, :, 0], q[:, :, 0])
    assert torch.allclose(rotated_q.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    assert not any(parameter.requires_grad for parameter in rope.parameters())
    with pytest.raises(ValueError, match="exceeds"):
        rope(torch.randn(1, 1, 5, 8), torch.randn(1, 1, 5, 8))
