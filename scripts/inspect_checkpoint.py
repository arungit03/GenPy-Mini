"""Inspect a valid checkpoint without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("checkpoint"); args = parser.parse_args()
    path = Path(args.checkpoint)
    if not (path / "COMPLETE").is_file(): raise SystemExit("invalid checkpoint: COMPLETE marker missing")
    print(json.dumps(json.loads((path / "metadata.json").read_text(encoding="utf-8")), indent=2))
    print(json.dumps(json.loads((path / "trainer_state.json").read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
