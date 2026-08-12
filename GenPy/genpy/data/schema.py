"""Standardized processed-document schema for the text-only Step 2 corpus."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class GenPyDocument:
    doc_id: str
    text: str
    content_hash: str
    source_dataset: Optional[str]
    source_config: Optional[str]
    source_url: Optional[str]
    source_dump: Optional[str]
    language: Optional[str]
    quality_score: Optional[float]
    char_count: int
    byte_count: int
    split: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "GenPyDocument":
        required = tuple(cls.__dataclass_fields__)
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"Document is missing required fields: {', '.join(missing)}")
        return cls(**{name: value[name] for name in required})
