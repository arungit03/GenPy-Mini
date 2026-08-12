import torch
from torch.nn import functional as F

from genpy.model import SwiGLU


def test_swiglu_formula_shapes_dimensions_and_gradients():
    torch.manual_seed(3)
    mlp = SwiGLU(hidden_size=16, intermediate_size=32)
    x = torch.randn(2, 5, 16, requires_grad=True)
    y = mlp(x)
    expected = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    assert y.shape == x.shape
    assert torch.allclose(y, expected)
    assert mlp.gate_proj.weight.shape == (32, 16)
    assert mlp.up_proj.weight.shape == (32, 16)
    assert mlp.down_proj.weight.shape == (16, 32)
    assert all(layer.bias is None for layer in (mlp.gate_proj, mlp.up_proj, mlp.down_proj))
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_swiglu_eval_is_deterministic_and_finite():
    mlp = SwiGLU(hidden_size=8, intermediate_size=16).eval()
    x = torch.randn(3, 8)
    assert torch.equal(mlp(x), mlp(x))
    assert torch.isfinite(mlp(x)).all()
