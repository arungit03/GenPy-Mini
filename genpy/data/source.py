"""Streaming source adapters; network access is intentionally lazy."""

import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from genpy.config import DataPipelineConfig


def load_source_rows(config: DataPipelineConfig, max_documents: Optional[int] = None) -> Iterable[dict]:
    """Load the configured Hugging Face source lazily and optionally cap rows."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to access the configured source."
        ) from exc
    try:
        rows = load_dataset(
            config.dataset.name,
            name=config.dataset.config,
            split=config.dataset.split,
            streaming=config.dataset.streaming,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to access configured dataset "
            f"{config.dataset.name}/{config.dataset.config} split={config.dataset.split} "
            f"streaming={config.dataset.streaming}: {exc}"
        ) from exc
    return itertools.islice(rows, max_documents) if max_documents is not None else rows


def load_jsonl_rows(path: Path) -> Iterator[dict]:
    """Yield local JSONL fixture rows without collecting the file in memory."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row {line_number} must be an object")
                yield row
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Local source file not found: {path}") from exc
