"""Run a bounded safe-fixture micro-overfit correctness test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.training.smoke import micro_overfit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    report = micro_overfit(
        args.model_config,
        args.data_config,
        maximum_steps=args.max_steps,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
