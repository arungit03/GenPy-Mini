"""Deterministic, indentation-preserving source normalization."""

from __future__ import annotations


class NormalizationError(ValueError):
    """A content rejection with a machine-readable reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def normalize_python_bytes(
    data: bytes,
    *,
    minimum_bytes: int,
    maximum_bytes: int,
    final_newline: bool = True,
) -> str:
    """Decode and normalize Python while preserving leading indentation."""
    if b"\0" in data:
        raise NormalizationError("null_byte")
    if len(data) > maximum_bytes:
        raise NormalizationError("too_large")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NormalizationError("invalid_utf8") from error
    control_count = sum(ord(char) < 9 or 13 < ord(char) < 32 for char in text)
    if text and control_count / len(text) > 0.01:
        raise NormalizationError("binary_file")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip(" \t") for line in text.split("\n"))
    text = text.rstrip("\n")
    if final_newline and text:
        text += "\n"
    if not text.strip() or len(text.encode("utf-8")) < minimum_bytes:
        raise NormalizationError("too_short")
    return text
