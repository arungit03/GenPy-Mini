"""Run the bounded CPU forward/backward smoke check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.training.smoke import smoke_forward  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-config", type=Path, default=Path("configs/data/smoke_packing.yaml")
    )
    args = parser.parse_args()
    print(json.dumps(smoke_forward(args.config, args.data_config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
