import torch

from genpy.model import RMSNorm


def test_rmsnorm_shape_dtype_finite_and_gradients():
    norm = RMSNorm(8)
    x = torch.randn(2, 3, 8, requires_grad=True)
    y = norm(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert norm.weight.shape == (8,)
    assert not hasattr(norm, "bias")


def test_rmsnorm_mixed_precision_returns_input_dtype():
    norm = RMSNorm(8)
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.randn(2, 8, dtype=dtype)
        y = norm(x)
        assert y.dtype == dtype
        assert torch.isfinite(y.float()).all()


def test_rmsnorm_matches_reference_formula():
    norm = RMSNorm(4, eps=1e-5)
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + norm.eps)
    assert torch.allclose(norm(x), expected)
