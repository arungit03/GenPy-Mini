"""Estimate packed output and temporary storage before writing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.training.packing import (  # noqa: E402
    estimate_production_storage,
    load_packing_config,
    prepare_packed_data,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_packing_config(args.config, ROOT)
    if config.packing["mode"] == "smoke":
        report = prepare_packed_data(config, dry_run=True)
    else:
        count_path = ROOT / "data/tokenizer/reports/exact_token_counts.json"
        counts = json.loads(count_path.read_text(encoding="utf-8"))
        token_count = sum(
            int(split.get("total_serialized_tokens", 0))
            for split in counts.get("splits", {}).values()
        )
        report = {
            **estimate_production_storage(config, token_count),
            "source_count_fingerprint": counts.get("tokenizer_fingerprint"),
            "production_tokenizer_fingerprint": config.tokenizer["fingerprint"],
            "status": "blocked_until_matching_production_token_counts_exist",
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
