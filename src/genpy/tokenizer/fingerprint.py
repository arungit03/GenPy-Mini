"""Checksums, fingerprints, and atomic metadata writes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash canonical JSON-compatible data."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def lines_fingerprint(lines: Iterable[str]) -> str:
    """Hash a deterministic stream of manifest lines."""
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Write formatted JSON through an atomic same-directory replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_checksum_file(directory: Path, filenames: Iterable[str]) -> dict[str, str]:
    """Write checksums for named artifact files, excluding the checksum file itself."""
    checksums = {name: sha256_file(directory / name) for name in sorted(filenames)}
    content = "".join(f"{digest}  {name}\n" for name, digest in checksums.items())
    temporary = directory / "checksums.sha256.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(directory / "checksums.sha256")
    return checksums


def read_checksum_file(path: Path) -> dict[str, str]:
    """Parse a standard two-space SHA-256 checksum file."""
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, filename = line.partition("  ")
        if not separator or len(digest) != 64 or not filename:
            raise ValueError("invalid checksum file")
        checksums[filename] = digest
    return checksums
