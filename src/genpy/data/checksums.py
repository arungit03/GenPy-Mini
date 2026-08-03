"""Streaming SHA-256 helpers and deterministic directory fingerprinting.

Every hashing function here reads files in bounded chunks so large archives
and repository snapshots never need to be loaded fully into memory -- this
matters on the 16 GB local development machine described in
``docs/development.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Final

from genpy.data.exceptions import ChecksumMismatchError

DEFAULT_CHUNK_SIZE: Final[int] = 1024 * 1024  # 1 MiB


def sha256_file(path: Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the lowercase hex SHA-256 digest of the file at ``path``.

    Reads the file in ``chunk_size``-byte chunks; never loads it fully into
    memory.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(stream: IO[bytes], *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the lowercase hex SHA-256 digest of an already-open binary stream.

    Consumes the stream from its current position to EOF.
    """
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_hex: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Verify that ``path`` hashes to ``expected_hex``; return the actual digest.

    Uses a constant-time comparison to avoid leaking timing information
    about how much of the digest matched. Raises
    :class:`ChecksumMismatchError` on mismatch.
    """
    actual = sha256_file(path, chunk_size=chunk_size)
    expected_normalized = expected_hex.strip().lower()
    if not hmac.compare_digest(actual, expected_normalized):
        raise ChecksumMismatchError(
            f"SHA-256 mismatch for {path.name}: expected {expected_normalized}, got {actual}"
        )
    return actual


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class FileFingerprint:
    """A single file's identity within a deterministic directory fingerprint."""

    relative_path: str
    size_bytes: int
    sha256: str


def fingerprint_directory(root: Path) -> tuple[FileFingerprint, ...]:
    """Return a deterministic, sorted fingerprint of every file under ``root``.

    Sorted by POSIX-style relative path so the result is stable across
    operating systems and filesystem iteration order.
    """
    fingerprints = [
        FileFingerprint(
            relative_path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in root.rglob("*")
        if path.is_file()
    ]
    return tuple(sorted(fingerprints, key=lambda item: item.relative_path))


def directory_digest(fingerprints: Sequence[FileFingerprint]) -> str:
    """Return a single deterministic SHA-256 digest summarizing a fingerprint.

    Two directory snapshots with identical file paths, sizes, and content
    hashes always produce the same digest, regardless of filesystem
    iteration order (the caller is expected to pass an already-sorted
    sequence, e.g. from :func:`fingerprint_directory`).
    """
    digest = hashlib.sha256()
    for item in fingerprints:
        digest.update(item.relative_path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(item.size_bytes).encode("ascii"))
        digest.update(b"\x00")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
