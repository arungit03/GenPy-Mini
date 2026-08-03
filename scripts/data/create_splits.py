"""Create deterministic splits through the idempotent Phase 2 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.data.pipeline import DatasetPipeline  # noqa: E402


def main() -> int:
    """Ensure split shards exist and print their counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report = DatasetPipeline(args.config, project_root=ROOT).build()
    print(json.dumps(report.get("split_counts", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
