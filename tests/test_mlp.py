import torch

from genpy.model import SwiGLU


def test_swiglu_equation_and_shapes() -> None:
    mlp = SwiGLU(8, 12)
    x = torch.randn(2, 3, 8)
    expected = mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    assert mlp(x).shape == (2, 3, 8)
    assert torch.allclose(mlp(x), expected)
    assert all(layer.bias is None for layer in (mlp.gate_proj, mlp.up_proj, mlp.down_proj))
