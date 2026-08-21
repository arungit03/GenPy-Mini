"""Instruction-document formatting and explicit BOS/EOS packing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from genpy.data.io import iter_records
from genpy.tokenizer import GenPyTokenizer


def format_training_document(record: dict) -> str:
    instruction = str(record.get("instruction", ""))
    input_text = str(record.get("input", ""))
    response = str(record.get("response", ""))
    if not instruction.strip() or not response.strip():
        raise ValueError("training records require non-empty instruction and response")
    parts = ["### User", instruction]
    if input_text:
        parts.extend(["", input_text])
    parts.extend(["", "### Assistant", response])
    return "\n".join(parts)


def encode_record(record: dict, tokenizer: GenPyTokenizer) -> list[int]:
    ids = [tokenizer.bos_token_id]
    ids.extend(tokenizer.encode(format_training_document(record)))
    ids.append(tokenizer.eos_token_id)
    if any(token_id < 0 or token_id >= 32000 for token_id in ids):
        raise ValueError("tokenizer produced an ID outside its vocabulary")
    return ids


def iter_encoded_records(path: str, tokenizer: GenPyTokenizer) -> Iterator[list[int]]:
    records = iter(iter_records(path))
    while True:
        batch = []
        for _ in range(256):
            try:
                batch.append(next(records))
            except StopIteration:
                break
        if not batch:
            return
        texts = [format_training_document(record) for record in batch]
        for encoded in tokenizer.encode_batch(texts):
            yield [tokenizer.bos_token_id, *encoded, tokenizer.eos_token_id]


def pack_records(records: Iterable[dict], tokenizer: GenPyTokenizer) -> list[int]:
    tokens: list[int] = []
    for record in records:
        tokens.extend(encode_record(record, tokenizer))
    return tokens
