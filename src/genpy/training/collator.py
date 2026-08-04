"""Fixed-shape packed sample collation."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from genpy.training.packed_dataset import PackedDataset, PackedSample
from genpy.training.sampler import DeterministicSampler


def collate_packed_samples(samples: list[PackedSample]) -> dict[str, Any]:
    """Stack packed tensors without changing labels or applying another shift."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    return {
        "input_ids": torch.stack([sample.input_ids for sample in samples]),
        "labels": torch.stack([sample.labels for sample in samples]),
        "attention_mask": torch.stack([sample.attention_mask for sample in samples]),
        "sample_indices": [sample.sample_index for sample in samples],
        "shard_ids": [sample.shard_id for sample in samples],
    }


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def build_packed_loader(
    dataset: PackedDataset,
    sampler: DeterministicSampler,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    drop_last: bool = False,
) -> DataLoader[Any]:
    """Build a deterministic Windows-safe packed-data loader."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch size and worker count are invalid")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_packed_samples,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=drop_last,
    )
