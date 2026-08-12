"""Console and JSONL training logger."""

import json
from pathlib import Path


class TrainingLogger:
    def __init__(self, log_dir: Path | str | None = None, name: str = "training") -> None:
        self.path = None if log_dir is None else Path(log_dir) / f"{name}.jsonl"
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        clean = {key: (float(value) if hasattr(value, "item") else value) for key, value in record.items()}
        print(" | ".join(f"{key}={value}" for key, value in clean.items()))
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(clean, sort_keys=True) + "\n")

    def close(self) -> None:
        return None
