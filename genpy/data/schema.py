"""Canonical, serializable schemas for Python instruction and code records."""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .registry import normalize_category, normalize_task_type


def _quality(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError("quality_score must satisfy 0.0 <= quality_score <= 1.0")
    return float(value)


@dataclass
class InstructionExample:
    id: str
    task_type: str
    category: str
    instruction: str
    response: str
    language: str = "python"
    input: str = ""
    source: str = "unknown"
    quality_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    family_id: str = ""
    syntax_valid: bool | None = None

    def validate(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("id must be a non-empty string")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if not isinstance(self.response, str) or not self.response.strip():
            raise ValueError("response must be a non-empty string")
        if self.language.strip().lower() != "python":
            raise ValueError("language must be python")
        normalize_task_type(self.task_type)
        normalize_category(self.category)
        _quality(self.quality_score)
        if not isinstance(self.input, str) or not isinstance(self.source, str):
            raise ValueError("input and source must be strings")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.syntax_valid is None:
            result.pop("syntax_valid")
        return result


@dataclass
class CodeExample:
    id: str
    task_type: str
    category: str
    code: str
    language: str = "python"
    source: str = "unknown"
    quality_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    family_id: str = ""
    syntax_valid: bool | None = None

    def validate(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("id must be a non-empty string")
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("code must be a non-empty string")
        if self.language.strip().lower() != "python":
            raise ValueError("language must be python")
        normalize_task_type(self.task_type)
        normalize_category(self.category)
        _quality(self.quality_score)
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.syntax_valid is None:
            result.pop("syntax_valid")
        return result


def example_from_mapping(raw: Mapping[str, Any]) -> InstructionExample | CodeExample:
    """Parse either canonical instruction data or raw code data."""
    if not isinstance(raw, Mapping):
        raise ValueError("each dataset record must be a JSON object")
    if "code" in raw and "instruction" not in raw and "response" not in raw:
        return CodeExample(
            id=str(raw.get("id", "")), task_type=str(raw.get("task_type", "code")),
            category=str(raw.get("category", "misc")), code=raw["code"],
            language=str(raw.get("language", "python")), source=str(raw.get("source", "unknown")),
            quality_score=raw.get("quality_score", 1.0), metadata=dict(raw.get("metadata", {})),
            family_id=str(raw.get("family_id", "")), syntax_valid=raw.get("syntax_valid"),
        )
    return InstructionExample(
        id=str(raw.get("id", "")), task_type=str(raw.get("task_type", "code_generation")),
        category=str(raw.get("category", "misc")), instruction=raw.get("instruction", ""),
        response=raw.get("response", ""), language=str(raw.get("language", "python")),
        input=str(raw.get("input", "")), source=str(raw.get("source", "unknown")),
        quality_score=raw.get("quality_score", 1.0), metadata=dict(raw.get("metadata", {})),
        family_id=str(raw.get("family_id", "")), syntax_valid=raw.get("syntax_valid"),
    )
