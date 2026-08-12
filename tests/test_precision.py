import pytest
import torch

from genpy.training.precision import PrecisionManager


def test_cpu_auto_and_fp32():
    assert PrecisionManager("auto", "cpu").mode == "fp32"
    assert PrecisionManager("fp32", "cpu").mode == "fp32"


def test_cpu_rejects_explicit_accelerator_precision():
    with pytest.raises(ValueError):
        PrecisionManager("bf16", "cpu")
    with pytest.raises(ValueError):
        PrecisionManager("fp16", "cpu")
