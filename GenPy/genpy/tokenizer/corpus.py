"""Incremental readers for Step 2 compressed text shards."""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence


@dataclass
class CorpusStats:
    documents: int = 0
    characters: int = 0
    utf8_bytes: int = 0

    def update(self, text: str) -> None:
        self.documents += 1
        self.characters += len(text)
        self.utf8_bytes += len(text.encode("utf-8"))

    def to_dict(self) -> Dict[str, int]:
        return {"documents": self.documents, "characters": self.characters, "utf8_bytes": self.utf8_bytes}


class CorpusReader:
    """Stream deterministic JSONL.GZ shards and collect consumption stats."""

    def __init__(self, input_dir: Path, pattern: str = "train-*.jsonl.gz", text_field: str = "text"):
        self.input_dir = Path(input_dir)
        self.pattern = pattern
        self.text_field = text_field
        self.stats = CorpusStats()

    def shard_paths(self) -> Sequence[Path]:
        paths = sorted(self.input_dir.glob(self.pattern), key=lambda path: path.name)
        if not paths:
            raise FileNotFoundError(f"No tokenizer corpus shards matched {self.input_dir / self.pattern}")
        return paths

    def rows(self) -> Iterator[dict]:
        for path in self.shard_paths():
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Malformed JSON in {path}:{line_number}") from exc
                        if not isinstance(row, dict):
                            raise ValueError(f"Expected JSON object in {path}:{line_number}")
                        yield row
            except (OSError, EOFError) as exc:
                raise ValueError(f"Unable to read compressed tokenizer shard {path}: {exc}") from exc

    def texts(self, max_documents: Optional[int] = None, max_bytes: Optional[int] = None) -> Iterator[str]:
        if max_documents is not None and max_documents < 0:
            raise ValueError("max_documents must be non-negative")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        for row in self.rows():
            if self.text_field not in row or not isinstance(row[self.text_field], str):
                raise ValueError(f"Tokenizer shard row is missing string field '{self.text_field}'")
            text = row[self.text_field]
            text_bytes = len(text.encode("utf-8"))
            if max_documents is not None and self.stats.documents >= max_documents:
                break
            if max_bytes is not None and self.stats.utf8_bytes + text_bytes > max_bytes:
                break
            self.stats.update(text)
            yield text


def iter_corpus_texts(input_dir: Path, pattern: str, text_field: str = "text", max_documents: Optional[int] = None, max_bytes: Optional[int] = None, stats: Optional[CorpusStats] = None) -> Iterator[str]:
    reader = CorpusReader(input_dir, pattern, text_field)
    if stats is not None:
        reader.stats = stats
    yield from reader.texts(max_documents, max_bytes)


def find_step2_manifest(manifest_dir: Path, explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Specified Step 2 manifest does not exist: {path}")
        return path
    paths = sorted(Path(manifest_dir).glob("prepare-*.json"), key=lambda path: path.name)
    return paths[-1] if paths else None


def inspect_corpus_gate(input_dir: Path, pattern: str, text_field: str, minimum_bytes: int, target_bytes: int) -> dict:
    """Measure a bounded corpus gate without labeling provenance as production."""
    reader = CorpusReader(input_dir, pattern, text_field)
    paths = reader.shard_paths()
    for _ in reader.texts():
        pass
    if reader.stats.utf8_bytes >= target_bytes:
        classification = "production_candidate_target_met"
    elif reader.stats.utf8_bytes >= minimum_bytes:
        classification = "production_candidate_minimum_met"
    else:
        classification = "development"
    return {
        "classification": classification,
        "train_shards": [path.name for path in paths],
        **reader.stats.to_dict(),
    }
