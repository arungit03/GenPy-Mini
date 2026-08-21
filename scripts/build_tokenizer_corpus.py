"""Build the train-only, boundary-preserving tokenizer corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.tokenizer.config import load_tokenizer_config
from genpy.tokenizer.corpus import collect_stats, write_corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer.yaml")
    parser.add_argument("--input", default="data/instruction/python/train.jsonl")
    parser.add_argument("--output", default="data/interim/tokenizer/train_corpus.jsonl")
    parser.add_argument("--stats", default="reports/tokenizer_corpus_stats.json")
    args = parser.parse_args()
    config = load_tokenizer_config(ROOT / args.config)
    source = ROOT / args.input
    output = ROOT / args.output
    stats = write_corpus(source, output, config)
    stats_path = ROOT / args.stats
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Tokenizer corpus: {output}")
    print(f"Training examples: {stats.training_examples:,}")
    print(f"Characters: {stats.characters:,}; bytes: {stats.bytes:,}; lines: {stats.lines:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
