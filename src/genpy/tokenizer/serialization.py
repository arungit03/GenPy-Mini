"""Canonical Phase 3 record serialization and control-token collision policy."""

from __future__ import annotations

from dataclasses import dataclass

from genpy.tokenizer.config import SPECIAL_TOKEN_TEXT

SYSTEM_MESSAGE = "You are GenPy, a Python code generator."


class SerializationError(ValueError):
    """Raised when content cannot be safely serialized."""


@dataclass(frozen=True, slots=True)
class SerializedRecord:
    """Serialized text and its structural-token count."""

    text: str
    structural_tokens: int


def reserved_token_collisions(text: str) -> tuple[str, ...]:
    """Return reserved controls found literally in ordinary content."""
    return tuple(token for token in SPECIAL_TOKEN_TEXT if token in text)


def validate_content(text: str, field_name: str) -> None:
    """Reject empty text and literal reserved-token collisions."""
    if not text:
        raise SerializationError(f"{field_name} must not be empty")
    collisions = reserved_token_collisions(text)
    if collisions:
        raise SerializationError(f"{field_name} contains a reserved control token")


def serialize_pretraining(code: str) -> SerializedRecord:
    """Serialize Python exactly, adding only the locked structural boundary."""
    validate_content(code, "code")
    return SerializedRecord(f"<|bos|><|code|>\n{code}<|end|><|eos|>\n", 4)


def serialize_instruction(
    prompt: str, code: str, system_message: str | None = None
) -> SerializedRecord:
    """Serialize one V1 prompt/code pair and omit the fixed system message."""
    validate_content(prompt, "prompt")
    validate_content(code, "code")
    if system_message is not None and system_message != SYSTEM_MESSAGE:
        raise SerializationError("instruction system message is not the fixed GenPy message")
    return SerializedRecord(
        f"<|bos|><|user|>\n{prompt}\n<|assistant|><|code|>\n{code}<|end|><|eos|>\n",
        6,
    )
