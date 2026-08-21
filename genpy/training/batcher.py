"""Deterministic random train windows and sequential validation windows."""

from __future__ import annotations

import torch

from .dataset import MemmapTokenDataset


class RandomWindowBatcher:
    def __init__(self, dataset: MemmapTokenDataset, batch_size: int, seed: int = 42) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, len(self.dataset), (self.batch_size,), generator=self.generator).tolist()
        pairs = [self.dataset[index] for index in starts]
        return torch.stack([pair[0] for pair in pairs]), torch.stack([pair[1] for pair in pairs])

    def state_dict(self) -> dict:
        return {"generator_state": self.generator.get_state(), "batch_size": self.batch_size}

    def load_state_dict(self, state: dict) -> None:
        self.generator.set_state(state["generator_state"])


class SequentialValidationBatcher:
    def __init__(self, dataset: MemmapTokenDataset, batch_size: int, batches: int | None = None) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        max_batches = len(dataset) // batch_size
        self.batches = min(max_batches, batches) if batches is not None else max_batches

    def __iter__(self):
        for batch_index in range(self.batches):
            starts = range(batch_index * self.batch_size, (batch_index + 1) * self.batch_size)
            pairs = [self.dataset[index] for index in starts]
            yield torch.stack([pair[0] for pair in pairs]), torch.stack([pair[1] for pair in pairs])
