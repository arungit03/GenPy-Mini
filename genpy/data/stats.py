"""Tokenizer-independent dataset statistics."""

from dataclasses import asdict, dataclass, field
from typing import Dict, Mapping


@dataclass
class DatasetStats:
    source_documents_seen: int = 0
    accepted_documents: int = 0
    rejected_documents: int = 0
    duplicate_documents: int = 0
    train_documents: int = 0
    validation_documents: int = 0
    total_characters: int = 0
    total_utf8_bytes: int = 0
    min_document_chars: int = 0
    max_document_chars: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def average_document_chars(self) -> float:
        if self.accepted_documents == 0:
            return 0.0
        return self.total_characters / self.accepted_documents

    def reject(self, reason: str) -> None:
        self.rejected_documents += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    def accept(self, split: str, char_count: int, byte_count: int) -> None:
        self.accepted_documents += 1
        if split == "train":
            self.train_documents += 1
        elif split == "validation":
            self.validation_documents += 1
        else:
            raise ValueError(f"Unknown split: {split}")
        self.total_characters += char_count
        self.total_utf8_bytes += byte_count
        if self.accepted_documents == 1:
            self.min_document_chars = char_count
            self.max_document_chars = char_count
        else:
            self.min_document_chars = min(self.min_document_chars, char_count)
            self.max_document_chars = max(self.max_document_chars, char_count)

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["average_document_chars"] = self.average_document_chars
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DatasetStats":
        field_names = tuple(cls.__dataclass_fields__)
        stats = cls(**{name: value.get(name, cls.__dataclass_fields__[name].default) for name in field_names})
        stats.rejection_reasons = dict(value.get("rejection_reasons", {}))
        return stats
