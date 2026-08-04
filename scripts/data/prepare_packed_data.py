"""Prepare or dry-run isolated deterministic packed data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.training.packing import load_packing_config, prepare_packed_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    config = load_packing_config(args.config, ROOT)
    if config.packing["mode"] != args.mode:
        raise SystemExit("--mode must match the versioned packing configuration")
    report = prepare_packed_data(
        config, dry_run=args.dry_run, force=args.force, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
