"""Lazy memory-mapped PyTorch dataset for packed GenPy shards."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from genpy.training.packed_format import (
    MASK_DTYPE,
    TOKEN_DTYPE,
    PackedShard,
    load_shard_metadata,
    validate_shard,
)


@dataclass(frozen=True, slots=True)
class PackedSample:
    """One input/target window and non-padding attention mask."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    sample_index: int
    shard_id: str


class PackedDataset(Dataset[PackedSample]):
    """Map global sample indices to lazily opened immutable binary shards."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        family: str,
        split: str,
        tokenizer_fingerprint: str,
        packing_configuration_hash: str,
    ) -> None:
        self.manifest_path = manifest_path
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("packing manifest must be an object")
        root = manifest_path.parent.parent
        shards: list[PackedShard] = []
        for relative in raw.get("shard_metadata", []):
            path = root / str(relative)
            metadata, _ = load_shard_metadata(path)
            if metadata.get("family") == family and metadata.get("split") == split:
                shards.append(
                    validate_shard(path, tokenizer_fingerprint, packing_configuration_hash)
                )
        self.shards = shards
        self.offsets: list[int] = []
        total = 0
        for shard in shards:
            total += shard.sample_count
            self.offsets.append(total)
        self._length = total
        self._token_maps: dict[int, np.memmap[Any, Any]] = {}
        self._mask_maps: dict[int, np.memmap[Any, Any]] = {}

    def __len__(self) -> int:
        return self._length

    def _mapping(self, shard_index: int) -> tuple[np.memmap[Any, Any], np.memmap[Any, Any]]:
        if shard_index not in self._token_maps:
            shard = self.shards[shard_index]
            self._token_maps[shard_index] = np.memmap(
                shard.tokens_path, dtype=TOKEN_DTYPE, mode="r",
                shape=(shard.sample_count, shard.stored_token_width),
            )
            self._mask_maps[shard_index] = np.memmap(
                shard.loss_mask_path, dtype=MASK_DTYPE, mode="r",
                shape=(shard.sample_count, shard.context_length),
            )
        return self._token_maps[shard_index], self._mask_maps[shard_index]

    def __getitem__(self, index: int) -> PackedSample:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.offsets, index)
        previous = self.offsets[shard_index - 1] if shard_index else 0
        local = index - previous
        tokens, masks = self._mapping(shard_index)
        row = np.asarray(tokens[local], dtype=np.int64).copy()
        active = np.asarray(masks[local], dtype=np.uint8).copy().astype(bool)
        input_ids = torch.from_numpy(row[:-1])
        labels = torch.from_numpy(row[1:]).clone()
        labels[~torch.from_numpy(active)] = -100
        return PackedSample(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(0),
            sample_index=index,
            shard_id=self.shards[shard_index].metadata_path.stem,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_token_maps"] = {}
        state["_mask_maps"] = {}
        return state
