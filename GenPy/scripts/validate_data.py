"""Validate processed compressed JSONL shards."""

import argparse
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from genpy.data.validation import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    report = validate_dataset(args.processed_dir, args.manifest)
    print(f"files: {len(report['files'])}")
    print(f"documents: {report['documents']}")
    print(f"valid: {report['valid']}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
