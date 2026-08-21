"""Build variable-length response-masked SFT token memmaps."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.data.io import iter_records, sha256_file
from genpy.tokenizer import GenPyTokenizer
from genpy.training.sft_dataset import encode_sft_record


def build_split(source: Path, output_dir: Path, tokenizer: GenPyTokenizer, sequence_length: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    input_path = output_dir / f"{stem}.input_ids.bin"
    labels_path = output_dir / f"{stem}.labels.bin"
    offsets_path = output_dir / f"{stem}.offsets.npy"
    input_tmp, labels_tmp, offsets_tmp = (Path(str(path) + ".tmp") for path in (input_path, labels_path, offsets_path))
    offsets = [0]
    stats = {"document_count": 0, "stored_token_count": 0, "assistant_token_count": 0, "ignored_prompt_token_count": 0, "truncation_count": 0, "max_record_tokens": 0}
    try:
        with input_tmp.open("wb") as input_handle, labels_tmp.open("wb") as labels_handle:
            for record in iter_records(source):
                encoding = encode_sft_record(record, tokenizer, sequence_length)
                np.asarray(encoding.input_ids, dtype=np.uint16).tofile(input_handle)
                np.asarray(encoding.labels, dtype=np.int32).tofile(labels_handle)
                offsets.append(offsets[-1] + len(encoding.input_ids))
                stats["document_count"] += 1
                stats["stored_token_count"] += len(encoding.input_ids)
                stats["assistant_token_count"] += sum(label != -100 for label in encoding.labels)
                stats["ignored_prompt_token_count"] += encoding.prompt_tokens
                stats["truncation_count"] += int(encoding.truncated)
                stats["max_record_tokens"] = max(stats["max_record_tokens"], len(encoding.input_ids))
        with offsets_tmp.open("wb") as offsets_handle:
            np.save(offsets_handle, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
        os.replace(input_tmp, input_path); os.replace(labels_tmp, labels_path); os.replace(offsets_tmp, offsets_path)
    finally:
        for temporary in (input_tmp, labels_tmp, offsets_tmp):
            if temporary.exists(): temporary.unlink()
    return {**stats, "source": str(source), "source_sha256": sha256_file(source), "input_ids_sha256": sha256_file(input_path), "labels_sha256": sha256_file(labels_path), "offsets_sha256": sha256_file(offsets_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/instruction/sft/train.jsonl")
    parser.add_argument("--validation", default="data/instruction/sft/validation.jsonl")
    parser.add_argument("--test", default="data/instruction/sft/test.jsonl")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--output-dir", default="data/instruction/tokenized")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--manifest", default="data/instruction/tokenized/SFT_TOKEN_CACHE_MANIFEST.json")
    args = parser.parse_args()
    tokenizer_dir = ROOT / args.tokenizer
    tokenizer = GenPyTokenizer.load(tokenizer_dir)
    if tokenizer.vocab_size != 32000:
        raise RuntimeError("SFT cache requires the exact 32K tokenizer")
    output_dir = ROOT / args.output_dir
    split_paths = {name: ROOT / value for name, value in {"train": args.train, "validation": args.validation, "test": args.test}.items()}
    stats = {name: build_split(path, output_dir, tokenizer, args.sequence_length) for name, path in split_paths.items()}
    manifest = {"format_version": 1, "tokenizer_name": tokenizer.name, "tokenizer_vocab_size": tokenizer.vocab_size, "tokenizer_manifest_sha256": sha256_file(tokenizer_dir / "TOKENIZER_MANIFEST.json"), "sequence_length": args.sequence_length, "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id, "test_split_immutable": True, "splits": stats}
    manifest_path = ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in value.items() if key in {"document_count", "stored_token_count", "assistant_token_count", "ignored_prompt_token_count", "truncation_count"}} for name, value in stats.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
