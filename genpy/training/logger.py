"""Append-only structured training metrics logger."""

from __future__ import annotations

import json
import time
from pathlib import Path


class MetricsLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict) -> None:
        record = {"timestamp": time.time(), **metrics}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
