"""Conservative, deterministic text normalization and quality filtering."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple

from genpy.config import ProcessingConfig


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    reason: Optional[str]
    char_count: int


def normalize_text(text: str, config: Optional[ProcessingConfig] = None) -> str:
    """Normalize text without flattening paragraphs or discarding valid Unicode."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if config is None:
        normalize_unicode = normalize_line_endings = remove_controls = normalize_whitespace = True
    else:
        normalize_unicode = config.normalize_unicode
        normalize_line_endings = config.normalize_line_endings
        remove_controls = config.remove_control_characters
        normalize_whitespace = config.normalize_whitespace
    value = unicodedata.normalize("NFC", text) if normalize_unicode else text
    if normalize_line_endings:
        value = value.replace("\r\n", "\n").replace("\r", "\n")
    if remove_controls:
        value = "".join(
            character for character in value
            if character in "\n\t" or unicodedata.category(character) != "Cc"
        )
    if normalize_whitespace:
        value = "\n".join(line.rstrip(" \t") for line in value.split("\n"))
        value = re.sub(r" {2,}", " ", value)
        value = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", value)
    return value.strip()


def assess_quality(text: object, min_chars: int, max_chars: int) -> QualityResult:
    """Return a transparent acceptance decision and rejection reason."""
    if text is None:
        return QualityResult(False, "missing_text", 0)
    if not isinstance(text, str):
        return QualityResult(False, "invalid_text_type", 0)
    if not text:
        return QualityResult(False, "empty_text", 0)
    length = len(text)
    if length < min_chars:
        return QualityResult(False, "too_short", length)
    if length > max_chars:
        return QualityResult(False, "too_long", length)
    return QualityResult(True, None, length)


def filter_text(text: object, min_chars: int, max_chars: int) -> Tuple[bool, str]:
    """Compatibility-friendly boolean/reason wrapper for quality filtering."""
    result = assess_quality(text, min_chars, max_chars)
    return result.accepted, result.reason or "accepted"
