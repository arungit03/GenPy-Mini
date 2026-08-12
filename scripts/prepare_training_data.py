"""Prepare one streaming cleaned-data split for Step 6 training."""

import argparse
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from genpy.training.tokenization import prepare_tokenized_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    metadata = prepare_tokenized_split(args.input_dir, args.tokenizer, args.output_dir, args.split, max_documents=args.max_documents, force=args.force)
    print(f"split: {metadata['split']}")
    print(f"documents: {metadata['document_count']}")
    print(f"tokens: {metadata['token_count']}")
    print(f"output: {args.output_dir / (args.split + '.bin')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
