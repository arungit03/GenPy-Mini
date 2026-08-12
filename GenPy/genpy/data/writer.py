"""Atomic compressed JSONL shard writing."""

import gzip
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, TextIO

from .schema import GenPyDocument


_SHARD_RE = re.compile(r"^(train|validation)-(\d{5})\.jsonl\.gz$")


class DocumentShardWriter:
    """Write bounded train and validation shards without holding them in RAM."""

    def __init__(self, output_dir: Path, shard_max_documents: int = 25000):
        if shard_max_documents <= 0:
            raise ValueError("shard_max_documents must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_max_documents = shard_max_documents
        self._handles: Dict[str, TextIO] = {}
        self._counts: Dict[str, int] = {"train": 0, "validation": 0}
        self._indexes = {
            "train": self._next_index("train"),
            "validation": self._next_index("validation"),
        }
        self._paths: List[str] = []
        self._closed = False
        for temporary in self.output_dir.glob("*.jsonl.gz.tmp"):
            temporary.unlink()

    def _next_index(self, split: str) -> int:
        indexes = []
        for path in self.output_dir.glob(f"{split}-*.jsonl.gz"):
            match = _SHARD_RE.match(path.name)
            if match:
                indexes.append(int(match.group(2)))
        return max(indexes, default=-1) + 1

    def _open(self, split: str) -> None:
        index = self._indexes[split]
        final_name = f"{split}-{index:05d}.jsonl.gz"
        temporary_path = self.output_dir / f"{final_name}.tmp"
        self._handles[split] = gzip.open(temporary_path, "wt", encoding="utf-8", newline="\n")
        self._counts[split] = 0

    def _finalize(self, split: str) -> None:
        handle = self._handles.pop(split, None)
        if handle is None:
            return
        handle.flush()
        handle.close()
        index = self._indexes[split]
        final_name = f"{split}-{index:05d}.jsonl.gz"
        temporary_path = self.output_dir / f"{final_name}.tmp"
        final_path = self.output_dir / final_name
        temporary_path.replace(final_path)
        self._paths.append(final_name)
        self._indexes[split] += 1
        self._counts[split] = 0

    def write(self, document: GenPyDocument) -> None:
        if self._closed:
            raise RuntimeError("Cannot write after the shard writer is closed")
        split = document.split
        if split not in self._handles:
            self._open(split)
        if self._counts[split] >= self.shard_max_documents:
            self._finalize(split)
            self._open(split)
        payload = json.dumps(document.to_dict(), ensure_ascii=False, sort_keys=True)
        self._handles[split].write(payload + "\n")
        self._handles[split].flush()
        self._counts[split] += 1

    def close(self) -> List[str]:
        if not self._closed:
            for split in tuple(self._handles):
                self._finalize(split)
            self._closed = True
        return list(self._paths)

    @property
    def shard_files(self) -> List[str]:
        return list(self._paths)

    def __enter__(self) -> "DocumentShardWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
