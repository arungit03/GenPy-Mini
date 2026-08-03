"""Reusable path-safety helpers for dataset acquisition.

Every function here is defensive by default: anything ambiguous is treated
as unsafe. These helpers are used both for local-directory copies and for
archive member extraction (zip/tar), where a hostile member path is a
classic "zip-slip" attack vector.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from genpy.data.exceptions import UnsafePathError

_NULL_BYTE = "\x00"
_DRIVE_LETTER_RE = re.compile(r"^[A-Za-z]:")


def validate_relative_member_path(member: str) -> PurePosixPath:
    """Validate an archive member path and return it as a safe relative ``PurePosixPath``.

    Rejects empty paths, null bytes, backslashes (ambiguous separators),
    absolute paths, Windows drive letters, and any ``..`` component.
    """
    if member is None or member.strip() == "":
        raise UnsafePathError("Archive member path is empty.")

    if _NULL_BYTE in member:
        raise UnsafePathError(f"Archive member path contains a null byte: {member!r}")

    if "\\" in member:
        raise UnsafePathError(f"Archive member path uses an unsafe backslash separator: {member!r}")

    if _DRIVE_LETTER_RE.match(member):
        raise UnsafePathError(f"Archive member path contains a drive letter: {member!r}")

    if member.startswith("/"):
        raise UnsafePathError(f"Archive member path is absolute: {member!r}")

    posix_path = PurePosixPath(member)

    if posix_path.is_absolute():  # pragma: no cover -- unreachable: "/" prefix is already rejected above
        raise UnsafePathError(f"Archive member path is absolute: {member!r}")

    if ".." in posix_path.parts:
        raise UnsafePathError(f"Archive member path escapes its root: {member!r}")

    if not posix_path.parts:
        raise UnsafePathError(f"Archive member path is empty: {member!r}")

    return posix_path


def ensure_within_root(candidate: Path, root: Path, *, context: str = "path") -> Path:
    """Resolve ``candidate`` and confirm it lies within ``root``.

    Returns the resolved candidate path. Raises :class:`UnsafePathError` if
    the resolved candidate falls outside the resolved root.
    """
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()

    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise UnsafePathError(
            f"{context} resolves outside its approved root: {candidate} is not within {root}"
        ) from exc

    return resolved_candidate


def resolve_safe_extraction_path(root: Path, member: str) -> Path:
    """Combine member-path validation with a root-containment check.

    Returns the destination path for extracting ``member`` under ``root``.
    Safe to use for both zip and tar members. Callers must still avoid ever
    creating symlink/hardlink entries during extraction -- see
    ``src/genpy/data/acquisition.py`` -- since a root-containment check
    alone cannot defend against a symlink planted by an earlier member.
    """
    relative = validate_relative_member_path(member)
    destination = root / Path(*relative.parts)
    return ensure_within_root(destination, root, context="archive member")


def is_safe_symlink(path: Path, root: Path) -> bool:
    """Return ``True`` if a symlink's resolved target stays within ``root``."""
    try:
        resolved_target = path.resolve()
        resolved_root = root.resolve()
    except OSError:
        return False

    try:
        resolved_target.relative_to(resolved_root)
    except ValueError:
        return False

    return True


def iter_safe_files(source_root: Path) -> Iterator[Path]:
    """Yield every regular file under ``source_root``, skipping unsafe symlinks.

    A symlink (file or directory) whose resolved target escapes
    ``source_root`` is never followed and never yielded. Directory symlinks
    that resolve safely within the root are followed; unsafe ones are
    pruned from the walk entirely. Uses an explicit stack rather than
    recursion so pathologically deep trees can't hit Python's recursion
    limit.
    """
    resolved_root = source_root.resolve()
    stack: list[Path] = [resolved_root]

    while stack:
        directory = stack.pop()
        for entry in sorted(directory.iterdir(), key=lambda p: p.name):
            if entry.is_symlink() and not is_safe_symlink(entry, resolved_root):
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                yield entry
