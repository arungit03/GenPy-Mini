"""Streaming JSON/JSONL input and atomic JSONL output helpers."""

import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

from .schema import CodeExample, InstructionExample, example_from_mapping

Example = InstructionExample | CodeExample


def iter_records(path: str | Path) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file or a JSON list/object."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = raw.get("examples", raw) if isinstance(raw, dict) else raw
        if not isinstance(records, list):
            raise ValueError("JSON input must be a list or an object with an 'examples' list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("JSON dataset entries must be objects")
            yield record
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            yield record


def load_examples(path: str | Path) -> list[Example]:
    return [example_from_mapping(record) for record in iter_records(path)]


def write_jsonl_atomic(path: str | Path, examples: list[Example]) -> str:
    """Write newline-terminated UTF-8 JSONL via a same-directory temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
