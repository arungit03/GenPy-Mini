import random

import numpy as np
import torch

from genpy.training.checkpoint import capture_rng_state, restore_rng_state


def test_python_numpy_torch_rng_restore() -> None:
    random.seed(4); np.random.seed(4); torch.manual_seed(4)
    state = capture_rng_state(); expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    restore_rng_state(state); actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert expected == actual
