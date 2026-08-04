"""Check Phase 2 readiness for smoke, candidate, or production tokenizer training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.tokenizer.config import load_tokenizer_config  # noqa: E402
from genpy.tokenizer.validation import check_readiness  # noqa: E402

RANK = {
    "NOT_READY": 0,
    "READY_FOR_SMOKE_TOKENIZER": 1,
    "READY_FOR_CANDIDATE_TOKENIZER": 2,
    "READY_FOR_PRODUCTION_TOKENIZER": 3,
}


def main() -> int:
    """Print readiness evidence and fail when the config's required status is unmet."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_tokenizer_config(args.config, ROOT)
    result = check_readiness(config)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    required = str(config.raw["readiness"]["required_status"])
    return 0 if RANK[result.status] >= RANK[required] else 1


if __name__ == "__main__":
    raise SystemExit(main())
