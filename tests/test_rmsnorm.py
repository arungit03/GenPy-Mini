import torch

from genpy.model import RMSNorm


def test_rmsnorm_matches_reference_and_has_no_bias() -> None:
    norm = RMSNorm(4, eps=1e-5)
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    assert torch.allclose(norm(x), expected)
    assert tuple(norm.weight.shape) == (4,)
    assert not hasattr(norm, "bias")
    assert torch.isfinite(norm(torch.randn(2, 3, 4) * 1e20)).all()
