"""Build train/validation uint16 token caches from the approved splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.data.io import sha256_file
from genpy.tokenizer import GenPyTokenizer
from genpy.training.packing import iter_encoded_records

EXPECTED_TRAIN_HASH = "17ba25f0154d1ffa04fdd4b91a22123a0770fe6aa76416ba57e4630264cb0b44"


def write_split(source: Path, target: Path, tokenizer: GenPyTokenizer) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    document_lengths, chunk = [], []
    token_count = bos_count = eos_count = 0
    with temp.open("wb") as handle:
        for document in iter_encoded_records(str(source), tokenizer):
            document_lengths.append(len(document)); token_count += len(document)
            bos_count += document.count(tokenizer.bos_token_id); eos_count += document.count(tokenizer.eos_token_id)
            chunk.extend(document)
            if len(chunk) >= 262144:
                np.asarray(chunk, dtype=np.uint16).tofile(handle); chunk.clear()
        if chunk: np.asarray(chunk, dtype=np.uint16).tofile(handle)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, target)
    return {"document_count": len(document_lengths), "token_count": token_count, "minimum_tokens_per_document": min(document_lengths, default=0), "maximum_tokens_per_document": max(document_lengths, default=0), "average_tokens_per_document": token_count / len(document_lengths) if document_lengths else 0.0, "bos_count": bos_count, "eos_count": eos_count, "bin_sha256": sha256_file(target)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/instruction/python/train.jsonl")
    parser.add_argument("--validation", default="data/instruction/python/validation.jsonl")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--output", default="data/tokenized/genpy-32k")
    args = parser.parse_args()
    train, validation, output = ROOT / args.train, ROOT / args.validation, ROOT / args.output
    train_hash = sha256_file(train)
    if train_hash != EXPECTED_TRAIN_HASH: raise RuntimeError(f"train dataset integrity failure: {train_hash} != {EXPECTED_TRAIN_HASH}")
    tokenizer = GenPyTokenizer.load(ROOT / args.tokenizer)
    if (tokenizer.vocab_size, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id) != (32000, 0, 1, 2, 3): raise RuntimeError("tokenizer compatibility failure")
    train_stats = write_split(train, output / "train.bin", tokenizer)
    validation_stats = write_split(validation, output / "validation.bin", tokenizer)
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.idx.json").write_text(json.dumps(train_stats, indent=2) + "\n", encoding="utf-8")
    (output / "validation.idx.json").write_text(json.dumps(validation_stats, indent=2) + "\n", encoding="utf-8")
    tokenizer_manifest = ROOT / args.tokenizer / "TOKENIZER_MANIFEST.json"
    manifest = {"format_version": 1, "tokenizer_name": tokenizer.name, "tokenizer_vocab_size": tokenizer.vocab_size, "tokenizer_hash": sha256_file(ROOT / args.tokenizer / "tokenizer.json"), "tokenizer_artifact_sha256": json.loads(tokenizer_manifest.read_text(encoding="utf-8")).get("artifact_sha256", "unknown"), "source_dataset_hash": train_hash, "validation_dataset_hash": sha256_file(validation), "train_token_count": train_stats["token_count"], "validation_token_count": validation_stats["token_count"], "train_document_count": train_stats["document_count"], "validation_document_count": validation_stats["document_count"], "dtype": "uint16", "vocab_size": tokenizer.vocab_size, "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id, "creation_timestamp": time.time(), "pipeline_version": "genpy-token-cache-v1", "train_bin_sha256": train_stats["bin_sha256"], "validation_bin_sha256": validation_stats["bin_sha256"]}
    (output / "TOKEN_CACHE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Train: {train_stats['document_count']:,} documents / {train_stats['token_count']:,} tokens")
    print(f"Validation: {validation_stats['document_count']:,} documents / {validation_stats['token_count']:,} tokens")
    print(f"Cache: {output}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
