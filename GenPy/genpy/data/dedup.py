"""Exact normalized-text deduplication."""

import hashlib
from typing import Set


def content_hash(text: str) -> str:
    """Return the lowercase SHA-256 digest of normalized text UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ExactDeduplicator:
    """Keep the first document for each exact content hash."""

    def __init__(self, hashes: Set[str] = None):
        self._hashes: Set[str] = set(hashes or ())

    def seen(self, digest: str) -> bool:
        return digest in self._hashes

    def add(self, digest: str) -> bool:
        """Add a digest and return True only when it was new."""
        if digest in self._hashes:
            return False
        self._hashes.add(digest)
        return True

    def accept(self, text: str) -> bool:
        return self.add(content_hash(text))

    @property
    def hashes(self) -> Set[str]:
        return set(self._hashes)
