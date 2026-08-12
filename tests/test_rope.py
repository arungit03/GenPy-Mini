import pytest
import torch

from genpy.model import RotaryEmbedding, apply_rotary_pos_emb


def test_rope_shapes_changes_qk_and_leaves_values_outside_helper():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    q = torch.randn(2, 4, 5, 8, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q)
    rotated_q, rotated_k = rope(q, k)
    assert rotated_q.shape == q.shape
    assert rotated_k.shape == k.shape
    assert not torch.allclose(rotated_q, q)
    assert not torch.allclose(rotated_k, k)
    cos, sin = rope.get_cos_sin(5)
    q2, k2 = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(rotated_q, q2)
    assert torch.allclose(rotated_k, k2)
    q3, k3 = apply_rotary_pos_emb(q, k, cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1))
    assert torch.allclose(rotated_q, q3)
    assert torch.allclose(rotated_k, k3)
    assert torch.equal(v, v.clone())


def test_rope_position_zero_is_identity_and_is_deterministic():
    rope = RotaryEmbedding(8, 8)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn_like(q)
    q_out, k_out = rope(q, k)
    assert torch.allclose(q_out, q)
    assert torch.allclose(k_out, k)
    assert torch.equal(rope(q, k)[0], rope(q, k)[0])


def test_rope_gradients_and_validation():
    rope = RotaryEmbedding(8, 8)
    q = torch.randn(1, 2, 4, 8, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    rope(q, k)[0].sum().backward()
    assert q.grad is not None and k.grad is None
    assert not list(rope.parameters())
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(7, 8)
    with pytest.raises(ValueError, match="exceeds"):
        rope(torch.randn(1, 1, 9, 8), torch.randn(1, 1, 9, 8))
