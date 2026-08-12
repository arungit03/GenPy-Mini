"""Streaming conversion of cleaned JSONL.GZ documents to uint16 tokens."""

import gzip
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from genpy.tokenizer.tokenizer import GenPyTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_shards(input_dir: Path, split: str) -> list[Path]:
    candidates = sorted(input_dir.glob(f"{split}*.jsonl.gz"))
    if not candidates:
        candidates = sorted(input_dir.glob(f"*{split}*.jsonl.gz"))
    if not candidates:
        raise FileNotFoundError(f"No {split} JSONL.GZ shards found in {input_dir}")
    return candidates


def _documents(shards: list[Path], text_field: str = "text") -> Iterator[tuple[str, Path]]:
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {shard}:{line_number}") from exc
                text = record.get(text_field) if isinstance(record, dict) else None
                if not isinstance(text, str):
                    raise ValueError(f"Missing string '{text_field}' in {shard}:{line_number}")
                yield text, shard


def prepare_tokenized_split(
    input_dir: Path,
    tokenizer_path: Path,
    output_dir: Path,
    split: str,
    *,
    max_documents: int | None = None,
    force: bool = False,
) -> dict:
    """Stream one cleaned split into an atomically-created uint16 token file."""
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive")
    input_dir = Path(input_dir)
    tokenizer_path = Path(tokenizer_path)
    output_dir = Path(output_dir)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.bin"
    metadata_path = output_dir / f"{split}_metadata.json"
    if (output_path.exists() or metadata_path.exists()) and not force:
        raise FileExistsError(f"Tokenized output exists; use force=True: {output_path}")
    tokenizer = GenPyTokenizer.from_file(tokenizer_path)
    contract = {
        "pad": tokenizer.pad_token_id,
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "unk": tokenizer.unk_token_id,
    }
    if tokenizer.vocab_size != 32000 or contract != {"pad": 0, "bos": 1, "eos": 2, "unk": 3}:
        raise ValueError(f"Tokenizer contract mismatch: vocab={tokenizer.vocab_size}, ids={contract}")
    shards = _source_shards(input_dir, split)
    temp_bin = output_dir / f".{split}.bin.{os.getpid()}.tmp"
    temp_meta = output_dir / f".{split}_metadata.{os.getpid()}.tmp"
    if temp_bin.exists():
        temp_bin.unlink()
    document_count = 0
    token_count = 0
    eos_count = 0
    try:
        with temp_bin.open("wb") as output:
            for text, _ in _documents(shards):
                ids = tokenizer.encode_document(text)
                if any(token_id < 0 or token_id >= tokenizer.vocab_size for token_id in ids):
                    raise ValueError("Tokenizer returned an out-of-range token ID")
                np.asarray(ids, dtype=np.uint16).tofile(output)
                document_count += 1
                token_count += len(ids)
                eos_count += 1
                if max_documents is not None and document_count >= max_documents:
                    break
        metadata = {
            "format_version": 1,
            "dtype": "uint16",
            "split": split,
            "token_count": token_count,
            "document_count": document_count,
            "source_shards": [str(path) for path in shards],
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "tokenizer_name": "GenPy-Tokenizer",
            "vocab_size": tokenizer.vocab_size,
            "special_token_ids": contract,
            "eos_count": eos_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        temp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp_bin, output_path)
        os.replace(temp_meta, metadata_path)
    except Exception:
        for path in (temp_bin, temp_meta):
            if path.exists():
                path.unlink()
        raise
    return metadata
