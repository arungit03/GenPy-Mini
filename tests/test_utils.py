import random

import numpy as np
import torch

from genpy.utils.device import get_device
from genpy.utils.seed import set_seed


def test_device_helper_returns_valid_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda"}
    if not torch.cuda.is_available():
        assert device.type == "cpu"


def test_seed_utility_is_reproducible():
    set_seed(42)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(42)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second


def test_seed_utility_executes_without_cuda():
    set_seed(7)
