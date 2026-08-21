import pytest
import torch

from genpy.verification.numerical import assert_finite, gradient_audit, is_finite_tensor


def test_nonfinite_detection() -> None:
    assert is_finite_tensor(torch.ones(3))
    assert not is_finite_tensor(torch.tensor([1.0, float("nan")]))
    with pytest.raises(ValueError, match="non-finite"):
        assert_finite(torch.tensor([float("inf")]))
