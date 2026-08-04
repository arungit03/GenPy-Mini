"""Stream exact tokenizer counts for all Phase 2 splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.tokenizer.config import load_tokenizer_config  # noqa: E402
from genpy.tokenizer.evaluation import count_corpus_tokens  # noqa: E402


def main() -> int:
    """Count with resumable per-shard state and no saved ID sequences."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    config = load_tokenizer_config(args.config, ROOT)
    report = count_corpus_tokens(
        config,
        args.artifact or config.artifact_path,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
