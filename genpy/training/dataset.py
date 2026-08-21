"""Memory-mapped uint16 token cache dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapTokenDataset(Dataset):
    def __init__(self, path: str | Path, sequence_length: int) -> None:
        self.path = Path(path)
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = sequence_length
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.tokens) <= sequence_length:
            raise ValueError("token cache must contain at least sequence_length + 1 tokens")

    def __len__(self) -> int:
        return len(self.tokens) - self.sequence_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        window = np.asarray(self.tokens[index:index + self.sequence_length + 1], dtype=np.int64).copy()
        values = torch.from_numpy(window)
        return values[:-1], values[1:]

    @property
    def total_tokens(self) -> int:
        return int(len(self.tokens))

    def window(self, start: int) -> np.ndarray:
        if start < 0 or start + self.sequence_length + 1 > len(self.tokens):
            raise IndexError("window exceeds token cache")
        return np.asarray(self.tokens[start:start + self.sequence_length + 1], dtype=np.int64).copy()

    def close(self) -> None:
        mmap = getattr(self.tokens, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
