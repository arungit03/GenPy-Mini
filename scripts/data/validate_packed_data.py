"""Validate checksums and contracts for every packed shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.training.packed_format import validate_packed_manifest  # noqa: E402
from genpy.training.packing import load_packing_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_packing_config(args.config, ROOT)
    report = validate_packed_manifest(
        config.output_root / "manifests/packing_manifest.json",
        str(config.tokenizer["fingerprint"]),
        config.config_hash,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
