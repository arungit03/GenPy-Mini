"""Device selection helper."""

import torch


def get_device() -> torch.device:
    """Return CUDA when it is available, otherwise return CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
