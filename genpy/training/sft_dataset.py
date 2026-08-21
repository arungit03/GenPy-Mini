"""Response-only supervised fine-tuning formatting, masking, and memmap data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from genpy.tokenizer import GenPyTokenizer
from genpy.training.packing import format_training_document


@dataclass(frozen=True)
class SFTEncoding:
    input_ids: list[int]
    labels: list[int]
    prompt_tokens: int
    assistant_tokens: int
    truncated: bool


def format_sft_document(record: dict) -> tuple[str, str]:
    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", ""))
    response = str(record.get("response", ""))
    if not instruction or not response.strip():
        raise ValueError("SFT records require non-empty instruction and response")
    user_record = {"instruction": instruction, "input": input_text, "response": response}
    full = format_training_document(user_record)
    marker = "\n\n### Assistant\n"
    prefix, assistant = full.split(marker, 1)
    return prefix + marker, assistant


def encode_sft_record(record: dict, tokenizer: GenPyTokenizer, sequence_length: int) -> SFTEncoding:
    prefix, assistant = format_sft_document(record)
    prompt_ids = [tokenizer.bos_token_id, *tokenizer.encode(prefix)]
    assistant_ids = [*tokenizer.encode(assistant), tokenizer.eos_token_id]
    capacity = sequence_length + 1
    truncated = len(prompt_ids) + len(assistant_ids) > capacity
    if truncated:
        if len(assistant_ids) >= capacity:
            assistant_ids = assistant_ids[-capacity:]
            prompt_ids = []
        else:
            prompt_capacity = capacity - len(assistant_ids)
            prompt_ids = prompt_ids[:1] + prompt_ids[-max(0, prompt_capacity - 1):]
    input_ids = prompt_ids + assistant_ids
    labels = [-100] * len(prompt_ids) + assistant_ids.copy()
    if len(input_ids) > capacity:
        raise AssertionError("SFT truncation exceeded sequence capacity")
    return SFTEncoding(input_ids, labels, len(prompt_ids), len(assistant_ids), truncated)


class SFTMemmapDataset(torch.utils.data.Dataset):
    """Variable-length token/label records backed by two memory maps."""

    def __init__(self, input_path: str | Path, labels_path: str | Path, offsets_path: str | Path, sequence_length: int, pad_token_id: int = 0) -> None:
        self.input_path = Path(input_path)
        self.labels_path = Path(labels_path)
        self.offsets = np.load(offsets_path, mmap_mode="r")
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self.inputs = np.memmap(self.input_path, mode="r", dtype=np.uint16)
        self.labels = np.memmap(self.labels_path, mode="r", dtype=np.int32)

    def __len__(self) -> int:
        return max(0, len(self.offsets) - 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        ids = np.asarray(self.inputs[start:end], dtype=np.int64)
        labels = np.asarray(self.labels[start:end], dtype=np.int64)
        x = np.full(self.sequence_length, self.pad_token_id, dtype=np.int64)
        y = np.full(self.sequence_length, -100, dtype=np.int64)
        usable = min(self.sequence_length, max(0, len(ids) - 1))
        if usable:
            x[:usable] = ids[:usable]
            y[:usable] = labels[1:usable + 1]
        return torch.from_numpy(x), torch.from_numpy(y)

    def close(self) -> None:
        for array in (self.inputs, self.labels, self.offsets):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class SFTRandomBatcher:
    def __init__(self, dataset: SFTMemmapDataset, batch_size: int, seed: int = 42) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(len(self.dataset), (self.batch_size,), generator=self.generator)
        batches = [self.dataset[int(index)] for index in indices]
        return torch.stack([item[0] for item in batches]), torch.stack([item[1] for item in batches])

    def state_dict(self) -> dict:
        return {"generator_state": self.generator.get_state().clone()}

    def load_state_dict(self, state: dict) -> None:
        self.generator.set_state(state["generator_state"])


class SFTSequentialBatcher:
    def __init__(self, dataset: SFTMemmapDataset, batch_size: int, batches: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.batches = batches
        self.cursor = 0

    def batches_iter(self):
        self.cursor = 0
        limit = min(len(self.dataset), self.batch_size * self.batches)
        while self.cursor < limit:
            indices = range(self.cursor, min(self.cursor + self.batch_size, limit))
            items = [self.dataset[index] for index in indices]
            self.cursor += len(items)
            yield torch.stack([item[0] for item in items]), torch.stack([item[1] for item in items])


def write_sft_manifest(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
