"""Memory-mapped packed token datasets and deterministic batch sampling."""

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class PackedTokenDataset(Dataset):
    """Return contiguous input/target windows from a uint16 token file."""

    def __init__(self, token_path: Path, sequence_length: int, metadata_path: Path | None = None) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.token_path = Path(token_path)
        if not self.token_path.is_file():
            raise FileNotFoundError(self.token_path)
        if metadata_path is None:
            metadata_path = self.token_path.with_name(f"{self.token_path.stem}_metadata.json")
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.is_file():
            raise FileNotFoundError(self.metadata_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("dtype") != "uint16":
            raise ValueError("token metadata dtype must be uint16")
        if self.metadata.get("token_count", -1) < 0:
            raise ValueError("token metadata token_count is invalid")
        self.sequence_length = sequence_length
        expected_bytes = int(self.metadata["token_count"]) * np.dtype(np.uint16).itemsize
        if self.token_path.stat().st_size != expected_bytes:
            raise ValueError("token file size does not match metadata token_count")
        self.tokens = np.memmap(self.token_path, dtype=np.uint16, mode="r")
        if self.tokens.size != self.metadata["token_count"]:
            raise ValueError("token file length does not match metadata")
        vocab_size = self.metadata.get("vocab_size")
        if vocab_size is not None and np.any(self.tokens >= int(vocab_size)):
            raise ValueError("token file contains an ID outside metadata vocabulary")
        self._length = max(0, (len(self.tokens) - 1) // sequence_length)

    def close(self) -> None:
        """Release the memory map, which is important before Windows cleanup."""
        mmap_handle = getattr(self.tokens, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    def __del__(self):
        try:
            self.close()
        except (AttributeError, ValueError):
            pass

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        window = np.asarray(self.tokens[start : start + self.sequence_length + 1], dtype=np.int64)
        if window.size != self.sequence_length + 1:
            raise IndexError(index)
        return {
            "input_ids": torch.from_numpy(window[:-1].copy()),
            "targets": torch.from_numpy(window[1:].copy()),
        }


class StatefulBatchSampler(Sampler[list[int]]):
    """Single-process deterministic shuffled batches with resumable position."""

    def __init__(self, dataset_size: int, batch_size: int, seed: int = 0, shuffle: bool = True, drop_last: bool = True) -> None:
        if dataset_size < 0 or batch_size <= 0:
            raise ValueError("dataset_size must be non-negative and batch_size positive")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.batch_position = 0

    def _batches(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist() if self.shuffle else list(range(self.dataset_size))
        batches = [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._batches()
        while self.batch_position < len(batches):
            batch = batches[self.batch_position]
            self.batch_position += 1
            yield batch
        self.epoch += 1
        self.batch_position = 0

    def __len__(self) -> int:
        return len(self._batches())

    def state_dict(self) -> dict:
        return {"dataset_size": self.dataset_size, "batch_size": self.batch_size, "seed": self.seed, "shuffle": self.shuffle, "drop_last": self.drop_last, "epoch": self.epoch, "batch_position": self.batch_position}

    def load_state_dict(self, state: dict) -> None:
        for key in ("dataset_size", "batch_size", "seed", "shuffle", "drop_last"):
            if getattr(self, key) != state[key]:
                raise ValueError(f"sampler mismatch for {key}")
        self.epoch = int(state["epoch"])
        self.batch_position = int(state["batch_position"])


def create_dataloader(dataset: Dataset, batch_size: int, *, seed: int = 0, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = False) -> tuple[DataLoader, StatefulBatchSampler]:
    sampler = StatefulBatchSampler(len(dataset), batch_size, seed=seed, shuffle=shuffle, drop_last=shuffle)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    return loader, sampler
