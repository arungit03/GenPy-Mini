"""Train-only corpus formatting and streaming statistics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from genpy.data.io import iter_records

from .config import TokenizerConfig


@dataclass
class CorpusStats:
    training_examples: int = 0
    characters: int = 0
    bytes: int = 0
    lines: int = 0
    total_instruction_characters: int = 0
    total_input_characters: int = 0
    total_response_characters: int = 0

    def to_dict(self) -> dict:
        result = asdict(self)
        count = self.training_examples or 1
        result["average_instruction_length"] = self.total_instruction_characters / count
        result["average_input_length"] = self.total_input_characters / count
        result["average_response_length"] = self.total_response_characters / count
        return result


def format_document(record: dict, config: TokenizerConfig) -> str:
    """Format one instruction row with explicit, stable document boundaries."""
    instruction = str(record.get("instruction", "")) if config.include_instruction else ""
    input_text = str(record.get("input", "")) if config.include_input else ""
    response = str(record.get("response", "")) if config.include_response else ""
    if not instruction or not response:
        raise ValueError("tokenizer corpus rows require instruction and response")
    parts = ["<BOS>", "### User", instruction]
    if input_text:
        parts.extend(["", input_text])
    parts.extend(["", "### Assistant", response, "<EOS>"])
    return "\n".join(parts)


def iter_documents(path: str | Path, config: TokenizerConfig, skip_invalid: bool = False) -> Iterator[str]:
    for record in iter_records(path):
        try:
            yield format_document(record, config)
        except ValueError:
            if not skip_invalid:
                raise


def collect_stats(path: str | Path, config: TokenizerConfig, skip_invalid: bool = False) -> CorpusStats:
    stats = CorpusStats()
    for record in iter_records(path):
        try:
            document = format_document(record, config)
        except ValueError:
            if not skip_invalid:
                raise
            continue
        stats.training_examples += 1
        stats.characters += len(document)
        stats.bytes += len(document.encode("utf-8"))
        stats.lines += document.count("\n") + 1
        stats.total_instruction_characters += len(str(record.get("instruction", "")))
        stats.total_input_characters += len(str(record.get("input", "")))
        stats.total_response_characters += len(str(record.get("response", "")))
    return stats


def write_corpus(path: str | Path, output: str | Path, config: TokenizerConfig) -> CorpusStats:
    """Write one JSON object containing only text per line; no dataset metadata."""
    stats = CorpusStats()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for document in iter_documents(path, config):
            handle.write(json.dumps({"text": document}, ensure_ascii=False) + "\n")
            stats.training_examples += 1
            stats.characters += len(document)
            stats.bytes += len(document.encode("utf-8"))
            stats.lines += document.count("\n") + 1
    return stats
