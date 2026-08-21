import random

import numpy as np
import torch

from genpy.utils.reproducibility import set_seed


def test_repeated_seeding_reproduces_python_numpy_and_torch() -> None:
    set_seed(42)
    first = (random.random(), np.random.rand(5), torch.rand(5))
    set_seed(42)
    second = (random.random(), np.random.rand(5), torch.rand(5))
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
