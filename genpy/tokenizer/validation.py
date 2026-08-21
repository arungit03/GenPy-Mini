"""Validation helpers for tokenizer contracts and metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .tokenizer import GenPyTokenizer


@dataclass
class TextMetrics:
    examples: int
    characters: int
    bytes: int
    tokens: int
    unknown_tokens: int

    def to_dict(self) -> dict:
        result = asdict(self)
        result["tokens_per_character"] = self.tokens / self.characters if self.characters else 0.0
        result["characters_per_token"] = self.characters / self.tokens if self.tokens else 0.0
        result["bytes_per_token"] = self.bytes / self.tokens if self.tokens else 0.0
        result["unknown_rate"] = self.unknown_tokens / self.tokens if self.tokens else 0.0
        result["average_tokens_per_example"] = self.tokens / self.examples if self.examples else 0.0
        return result


def measure_texts(tokenizer: GenPyTokenizer, texts: list[str] | tuple[str, ...]) -> TextMetrics:
    characters = sum(len(text) for text in texts)
    byte_count = sum(len(text.encode("utf-8")) for text in texts)
    token_count = 0
    unknown = 0
    for text in texts:
        ids = tokenizer.encode(text)
        token_count += len(ids)
        unknown += sum(token_id == tokenizer.unk_token_id for token_id in ids)
    return TextMetrics(len(texts), characters, byte_count, token_count, unknown)


def contract_checks(tokenizer: GenPyTokenizer) -> dict[str, bool]:
    return {
        "vocabulary_size": tokenizer.vocab_size == 32000,
        "special_token_ids": (
            tokenizer.pad_token_id == 0
            and tokenizer.bos_token_id == 1
            and tokenizer.eos_token_id == 2
            and tokenizer.unk_token_id == 3
        ),
    }
