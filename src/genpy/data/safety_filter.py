"""Static unsafe-code topic filters; downloaded code is never executed."""

from __future__ import annotations

import re

UNSAFE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "credential_theft",
        re.compile(r"(?i)\b(?:steal|dump|harvest).{0,30}(?:credential|cookie|token)"),
    ),
    ("keylogging", re.compile(r"(?i)\b(?:keylogger|keyboard\s+hook|pynput\.keyboard)\b")),
    ("ransomware", re.compile(r"(?i)\b(?:ransomware|encrypt.{0,20}files.{0,20}bitcoin)\b")),
    (
        "persistence_or_evasion",
        re.compile(r"(?i)\b(?:disable_defender|uac_bypass|startup_registry)\b"),
    ),
    ("malware_delivery", re.compile(r"(?i)\b(?:dropper|payload_delivery|reverse_shell)\b")),
    ("exploit_automation", re.compile(r"(?i)\b(?:exploit_all|mass_exploit|autopwn)\b")),
)


def scan_unsafe_content(text: str) -> tuple[str, ...]:
    """Return unsafe-code categories using conservative static patterns."""
    return tuple(sorted(category for category, pattern in UNSAFE_PATTERNS if pattern.search(text)))
