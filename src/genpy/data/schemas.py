"""Typed record schemas and stable identifiers for GenPy datasets."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SplitName = Literal["train", "validation", "test"]
Role = Literal["system", "user", "assistant"]


def content_sha256(text: str) -> str:
    """Return the SHA-256 digest of normalized UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_record_id(source_id: str, identity: str, digest: str) -> str:
    """Create a deterministic identifier without exposing record content."""
    payload = f"{source_id}\0{identity}\0{digest}".encode()
    return f"genpy-{hashlib.sha256(payload).hexdigest()[:24]}"


@dataclass(slots=True)
class QualityInfo:
    """Static validation and quality metadata for a pretraining record."""

    ast_valid: bool
    secret_scan_passed: bool
    pii_scan_passed: bool
    quality_score: float

    def validate(self) -> None:
        """Validate quality metadata bounds."""
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be between 0.0 and 1.0")


@dataclass(slots=True)
class PretrainingRecord:
    """One provenance-bearing Python pretraining record."""

    record_id: str
    text: str
    language: str
    source_id: str
    source_url: str
    repository: str | None
    revision: str
    file_path: str
    licence_spdx: str
    content_sha256: str
    generation_method: str
    quality: QualityInfo
    split_group: str
    split: SplitName = "train"

    def validate(self) -> None:
        """Validate mandatory fields and content integrity."""
        required = {
            "record_id": self.record_id,
            "text": self.text,
            "language": self.language,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "revision": self.revision,
            "file_path": self.file_path,
            "licence_spdx": self.licence_spdx,
            "generation_method": self.generation_method,
            "split_group": self.split_group,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"missing mandatory pretraining fields: {', '.join(missing)}")
        if self.language.lower() != "python":
            raise ValueError("pretraining language must be python")
        if self.content_sha256 != content_sha256(self.text):
            raise ValueError("content_sha256 does not match text")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split: {self.split}")
        self.quality.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to JSON-serializable data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PretrainingRecord:
        """Construct and validate a record from decoded JSON."""
        data = dict(value)
        quality = data.get("quality")
        if not isinstance(quality, dict):
            raise ValueError("quality must be an object")
        data["quality"] = QualityInfo(**quality)
        record = cls(**data)
        record.validate()
        return record


@dataclass(slots=True)
class Message:
    """One instruction conversation message."""

    role: Role
    content: str

    def validate(self) -> None:
        """Reject unknown roles and empty messages."""
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"unknown message role: {self.role}")
        if not self.content.strip():
            raise ValueError("message content must not be empty")


@dataclass(slots=True)
class InstructionRecord:
    """One V1 natural-language-to-Python instruction record."""

    record_id: str
    messages: list[Message]
    category: str
    difficulty: str
    tests: list[dict[str, Any]] = field(default_factory=list)
    source_id: str = ""
    licence_spdx: str = ""
    content_sha256: str = ""
    generation_method: str = "human"
    problem_family: str = ""
    split: SplitName = "train"

    def validate(self) -> None:
        """Validate the V1 instruction shape and assistant Python syntax."""
        required = {
            "record_id": self.record_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "source_id": self.source_id,
            "licence_spdx": self.licence_spdx,
            "generation_method": self.generation_method,
            "problem_family": self.problem_family,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"missing mandatory instruction fields: {', '.join(missing)}")
        for message in self.messages:
            message.validate()
        assistants = [message for message in self.messages if message.role == "assistant"]
        if len(assistants) != 1:
            raise ValueError("V1 instruction records require exactly one assistant answer")
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("instruction record requires a user message")
        try:
            ast.parse(assistants[0].content, feature_version=(3, 11))
        except SyntaxError as error:
            raise ValueError("assistant answer is not valid Python 3.11 syntax") from error
        canonical = _canonical_messages(self.messages)
        if self.content_sha256 != content_sha256(canonical):
            raise ValueError("content_sha256 does not match canonical messages")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split: {self.split}")
        json.dumps(self.tests, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to JSON-serializable data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InstructionRecord:
        """Construct and validate a record from decoded JSON."""
        data = dict(value)
        raw_messages = data.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("messages must be a list")
        data["messages"] = [Message(**message) for message in raw_messages]
        record = cls(**data)
        record.validate()
        return record


def _canonical_messages(messages: list[Message]) -> str:
    return json.dumps(
        [asdict(message) for message in messages],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def instruction_content_digest(messages: list[Message]) -> str:
    """Hash instruction messages using the schema's canonical representation."""
    return content_sha256(_canonical_messages(messages))
