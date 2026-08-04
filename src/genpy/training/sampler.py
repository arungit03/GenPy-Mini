"""Deterministic epoch, rank, and resume-aware packed-data sampler."""

from __future__ import annotations

from collections.abc import Iterator, Sized
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True, slots=True)
class SamplerState:
    """Exact deterministic resume position."""

    epoch: int
    cursor: int


class DeterministicSampler(Sampler[int]):
    """Stable shuffled indices separated across distributed ranks."""

    def __init__(
        self,
        data_source: Sized,
        *,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
        state: SamplerState | None = None,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed rank or world size")
        self.data_source = data_source
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.epoch = state.epoch if state else 0
        self.cursor = state.cursor if state else 0

    def _indices(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        if self.drop_last:
            indices = indices[: len(indices) - len(indices) % self.world_size]
        return indices[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[int]:
        yield from self._indices()[self.cursor :]

    def __len__(self) -> int:
        return max(0, len(self._indices()) - self.cursor)

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic epoch and reset its cursor."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch
        self.cursor = 0

    def advance(self, count: int) -> None:
        """Advance the serializable sample cursor."""
        if count < 0 or self.cursor + count > len(self._indices()):
            raise ValueError("sampler cursor advance is invalid")
        self.cursor += count

    def state_dict(self) -> dict[str, int]:
        """Return exact JSON-safe resume state."""
        return asdict(SamplerState(self.epoch, self.cursor))
