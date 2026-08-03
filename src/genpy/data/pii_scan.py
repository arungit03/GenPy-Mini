"""Conservative scanners for common personal-data forms."""

from __future__ import annotations

import ipaddress
import re

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)")
IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
PERSONAL_IDENTIFIER = re.compile(r"(?i)\b(?:ssn|social_security_number|national_id)\s*[:=]")


def scan_pii(text: str) -> tuple[str, ...]:
    """Return PII categories only and avoid retaining matched values."""
    categories: set[str] = set()
    if EMAIL.search(text):
        categories.add("email_address")
    if PHONE.search(text):
        categories.add("phone_number")
    if PERSONAL_IDENTIFIER.search(text):
        categories.add("personal_identifier")
    for match in IPV4.finditer(text):
        try:
            address = ipaddress.ip_address(match.group(0))
        except ValueError:
            continue
        if address.is_private:
            categories.add("private_ip_address")
            break
    return tuple(sorted(categories))
