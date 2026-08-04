from __future__ import annotations

import torch

from genpy.model.norm import RMSNorm


def test_rmsnorm_matches_reference_and_has_no_bias() -> None:
    inputs = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    norm = RMSNorm(4, 1e-5)
    expected = inputs * torch.rsqrt(inputs.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(norm(inputs), expected)
    assert list(dict(norm.named_parameters())) == ["weight"]
