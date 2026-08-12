"""Inspect IDs, tokens, and round-trip behavior for a tokenizer artifact."""

import argparse
import sys
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root
ensure_project_root()

from genpy.tokenizer.tokenizer import GenPyTokenizer


def safe(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    tokenizer = GenPyTokenizer.from_file(args.tokenizer)
    ids = tokenizer.encode(args.text)
    decoded = tokenizer.decode(ids)
    print(f"original text: {safe(args.text)}")
    print(f"token IDs: {ids}")
    print(f"token strings: {[safe(token) for token in tokenizer.token_strings(ids)]}")
    print(f"token count: {len(ids)}")
    print(f"decoded text: {safe(decoded)}")
    print(f"round-trip success: {decoded == args.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
