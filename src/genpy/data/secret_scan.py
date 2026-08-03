"""Conservative secret-pattern scanning that never returns matched values."""

from __future__ import annotations

import math
import re
from collections import Counter

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    (
        "password_assignment",
        re.compile(r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"\n]{4,}['\"]"),
    ),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("generic_api_key", re.compile(r"(?i)\bapi[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]")),
    (
        "access_token",
        re.compile(r"(?i)\b(?:access|auth)[_-]?token\s*[:=]\s*['\"][^'\"\n]{16,}['\"]"),
    ),
    ("connection_string", re.compile(r"(?i)\b(?:postgres|mysql|mongodb(?:\+srv)?)://[^\s'\"]+")),
    ("cookie", re.compile(r"(?i)\b(?:session|auth)[_-]?cookie\s*[:=]")),
    ("dotenv_content", re.compile(r"(?m)^(?:SECRET|TOKEN|PASSWORD|API_KEY)=")),
)
HIGH_ENTROPY_CANDIDATE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9+/=_-]{32,}(?![A-Za-z0-9])")


def _entropy(value: str) -> float:
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def scan_secrets(text: str) -> tuple[str, ...]:
    """Return detection categories only; secret-like text is never returned."""
    categories = {category for category, pattern in PATTERNS if pattern.search(text)}
    for match in HIGH_ENTROPY_CANDIDATE.finditer(text):
        candidate = match.group(0)
        if any(char.isalpha() for char in candidate) and any(char.isdigit() for char in candidate):
            if _entropy(candidate) >= 4.3:
                categories.add("high_entropy_string")
                break
    return tuple(sorted(categories))
