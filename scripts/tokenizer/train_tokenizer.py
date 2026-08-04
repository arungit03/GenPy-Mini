"""Train a versioned GenPy byte-level BPE tokenizer on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.tokenizer.config import load_tokenizer_config  # noqa: E402
from genpy.tokenizer.trainer import train_tokenizer  # noqa: E402
from genpy.tokenizer.validation import check_readiness  # noqa: E402


def main() -> int:
    """Train after readiness and replacement-safety checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_tokenizer_config(args.config, ROOT)
    if args.seed is not None and args.seed != int(config.tokenizer["training_seed"]):
        raise SystemExit("--seed must match the versioned tokenizer configuration")
    if args.workers != 1:
        raise SystemExit("Phase 3 deterministic training currently requires --workers 1")
    if args.dry_run:
        print(json.dumps(check_readiness(config).to_dict(), indent=2, sort_keys=True))
        return 0
    metadata = train_tokenizer(
        config,
        mode=args.mode,
        force=args.force,
        maximum_bytes=args.max_bytes,
        maximum_records=args.max_records,
        source_ids=set(args.sources) if args.sources else None,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output or config.artifact_path),
                "status": metadata["status"],
                "vocabulary_size": metadata["actual_vocabulary_size"],
                "fingerprint": metadata["tokenizer_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
