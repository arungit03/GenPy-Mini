"""Estimate later training memory without allocating a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.model.config import load_model_config  # noqa: E402
from genpy.model.memory import estimate_training_memory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    args = parser.parse_args()
    config = load_model_config(args.config, ROOT)
    report = estimate_training_memory(config, args.sequence_length, args.micro_batch_size)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
