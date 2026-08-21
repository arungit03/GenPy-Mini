import torch

from genpy.verification.precision import choose_precision


def test_precision_selection_is_cpu_safe() -> None:
    choice = choose_precision("auto", "cpu")
    assert choice.mode == "fp32"
    assert choice.dtype == torch.float32
    assert choice.supported


def test_explicit_fp32_selection() -> None:
    choice = choose_precision("fp32", "cpu")
    assert choice.mode == "fp32" and choice.supported
