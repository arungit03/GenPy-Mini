"""Verify and checksum a complete tokenizer artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.tokenizer.evaluation import package_artifact  # noqa: E402
from genpy.tokenizer.tokenizer import GenPyTokenizer  # noqa: E402


def main() -> int:
    """Refresh checksums and validate the packaged result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    checksums = package_artifact(args.artifact)
    tokenizer = GenPyTokenizer.load(args.artifact)
    print(
        json.dumps(
            {
                "artifact": str(args.artifact),
                "checksummed_files": len(checksums),
                "fingerprint": tokenizer.fingerprint,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
