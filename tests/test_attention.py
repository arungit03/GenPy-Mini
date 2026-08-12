import torch

from genpy.model import GenPyAttention


def test_attention_shapes_gradients_and_bias_free_projections():
    attention = GenPyAttention(
        hidden_size=32, num_heads=4, head_dim=8, max_seq_len=16, attention_dropout=0.0
    )
    x = torch.randn(2, 7, 32, requires_grad=True)
    y = attention(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None
    assert attention.num_heads == 4
    assert attention.head_dim == 8
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert getattr(attention, name).bias is None


def test_attention_is_causal():
    torch.manual_seed(4)
    attention = GenPyAttention(hidden_size=32, num_heads=4, head_dim=8, max_seq_len=8)
    attention.eval()
    prefix = torch.randn(1, 3, 32)
    first = torch.cat((prefix, torch.zeros(1, 1, 32)), dim=1)
    second = torch.cat((prefix, torch.full((1, 1, 32), 100.0)), dim=1)
    with torch.no_grad():
        first_output = attention(first)
        second_output = attention(second)
    assert torch.allclose(first_output[:, :3], second_output[:, :3], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(first_output[:, 3], second_output[:, 3])
