"""Inspect one local GenPy tokenizer encoding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from genpy.tokenizer.tokenizer import GenPyTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    tokenizer = GenPyTokenizer.load(ROOT / args.tokenizer)
    ids = tokenizer.encode(args.text)
    tokens = tokenizer.token_strings(ids)
    decoded = tokenizer.decode(ids)
    print(f"Original text: {args.text!r}")
    print(f"Token IDs: {ids}")
    print(f"Token strings: {tokens}")
    print(f"Decoded text: {decoded!r}")
    print(f"Token count: {len(ids)}")
    print(f"Round-trip: {'PASS' if decoded == args.text else 'FAIL'}")
    return 0 if decoded == args.text else 1


if __name__ == "__main__":
    raise SystemExit(main())
