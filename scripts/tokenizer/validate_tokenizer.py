"""Validate checksums, metadata, special tokens, and byte-level round trips."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.tokenizer.tokenizer import GenPyTokenizer  # noqa: E402


def main() -> int:
    """Load and validate an artifact, returning nonzero on corruption."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = GenPyTokenizer.load(args.artifact)
    result = tokenizer.validate()
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
