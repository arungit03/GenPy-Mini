"""Tests for streaming SHA-256 and deterministic directory fingerprinting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from genpy.data.checksums import (
    directory_digest,
    fingerprint_directory,
    sha256_file,
    verify_sha256,
)
from genpy.data.exceptions import ChecksumMismatchError


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    file_path = tmp_path / "data.bin"
    file_path.write_bytes(b"hello world")
    assert sha256_file(file_path) == hashlib.sha256(b"hello world").hexdigest()


def test_sha256_file_handles_large_streamed_file(tmp_path: Path) -> None:
    file_path = tmp_path / "large.bin"
    chunk = b"x" * 1024
    with file_path.open("wb") as handle:
        for _ in range(5000):  # ~5 MiB, larger than the default 1 MiB chunk size
            handle.write(chunk)

    expected = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            expected.update(block)

    assert sha256_file(file_path, chunk_size=4096) == expected.hexdigest()


def test_verify_sha256_succeeds_for_correct_digest(tmp_path: Path) -> None:
    file_path = tmp_path / "data.bin"
    file_path.write_bytes(b"correct")
    expected = hashlib.sha256(b"correct").hexdigest()
    assert verify_sha256(file_path, expected) == expected


def test_verify_sha256_rejects_incorrect_digest(tmp_path: Path) -> None:
    file_path = tmp_path / "data.bin"
    file_path.write_bytes(b"actual content")
    wrong_digest = hashlib.sha256(b"different content").hexdigest()
    with pytest.raises(ChecksumMismatchError):
        verify_sha256(file_path, wrong_digest)


def test_verify_sha256_is_case_insensitive(tmp_path: Path) -> None:
    file_path = tmp_path / "data.bin"
    file_path.write_bytes(b"correct")
    expected = hashlib.sha256(b"correct").hexdigest().upper()
    assert verify_sha256(file_path, expected) == expected.lower()


def test_fingerprint_directory_is_deterministically_sorted(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_bytes(b"b")
    (tmp_path / "a.py").write_bytes(b"a")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.py").write_bytes(b"c")

    fingerprints = fingerprint_directory(tmp_path)
    paths = [item.relative_path for item in fingerprints]
    assert paths == sorted(paths)
    assert paths == ["a.py", "b.py", "nested/c.py"]


def test_directory_digest_is_stable_for_identical_content(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    for directory in (dir_a, dir_b):
        directory.mkdir()
        (directory / "file.py").write_bytes(b"identical content")

    digest_a = directory_digest(fingerprint_directory(dir_a))
    digest_b = directory_digest(fingerprint_directory(dir_b))
    assert digest_a == digest_b


def test_directory_digest_changes_when_content_changes(tmp_path: Path) -> None:
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / "file.py").write_bytes(b"version one")
    digest_before = directory_digest(fingerprint_directory(directory))

    (directory / "file.py").write_bytes(b"version two")
    digest_after = directory_digest(fingerprint_directory(directory))

    assert digest_before != digest_after


def test_directory_digest_changes_when_a_file_is_added(tmp_path: Path) -> None:
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / "file.py").write_bytes(b"content")
    digest_before = directory_digest(fingerprint_directory(directory))

    (directory / "extra.py").write_bytes(b"more")
    digest_after = directory_digest(fingerprint_directory(directory))

    assert digest_before != digest_after
