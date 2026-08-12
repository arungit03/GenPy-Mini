"""Evaluate a tokenizer on Step 2 validation shards."""

import argparse
import json
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root
ensure_project_root()

from genpy.tokenizer.corpus import CorpusReader
from genpy.tokenizer.evaluation import evaluate_texts
from genpy.tokenizer.tokenizer import GenPyTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--input-dir", default="data/processed", type=Path)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tokenizer = GenPyTokenizer.from_file(args.tokenizer)
    reader = CorpusReader(args.input_dir, "validation-*.jsonl.gz")
    try:
        metrics = evaluate_texts(tokenizer, reader.texts(max_documents=args.max_documents))
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No Step 2 validation shards are available; tokenizer evaluation cannot be fabricated."
        ) from exc
    metrics["source_shards"] = [path.name for path in reader.shard_paths()]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
