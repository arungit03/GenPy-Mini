"""Audit configured data sources without downloading content."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.data.licences import LicencePolicy  # noqa: E402
from genpy.data.source_registry import SourceRegistry  # noqa: E402


def main() -> int:
    """Print source audit JSON and return nonzero only for malformed configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--licences", type=Path, default=ROOT / "configs/data/licenses.yaml")
    args = parser.parse_args()
    registry = SourceRegistry.from_yaml(args.config)
    policy = LicencePolicy.from_yaml(args.licences)
    print(json.dumps(registry.audit(set(policy.allowlist)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
