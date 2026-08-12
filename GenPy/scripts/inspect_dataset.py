"""Inspect a tiny source sample without logging full documents."""

import argparse
import json
import sys
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from genpy.config import load_data_config
from genpy.data.source import load_jsonl_rows, load_source_rows


def _console_safe(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--source-jsonl", type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    config = load_data_config(args.config)
    rows = load_jsonl_rows(args.source_jsonl) if args.source_jsonl else load_source_rows(config)
    print(f"dataset name: {config.dataset.name}")
    print(f"dataset config: {config.dataset.config}")
    print(f"streaming status: {config.dataset.streaming}")
    inspected = 0
    for row in rows:
        if inspected >= args.limit:
            break
        if not isinstance(row, dict):
            continue
        if inspected == 0:
            print(f"available fields: {', '.join(sorted(row))}")
        text = row.get(config.dataset.text_field)
        preview = str(text).replace("\n", " ")[:240]
        metadata = [key for key in row if key != config.dataset.text_field]
        print(f"document {inspected + 1} text length: {len(text) if isinstance(text, str) else 'invalid'}")
        print(f"metadata fields present: {', '.join(sorted(metadata)) or 'none'}")
        print(f"preview: {_console_safe(preview)}")
        inspected += 1
    print(f"number inspected: {inspected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
