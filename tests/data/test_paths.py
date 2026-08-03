"""Tests for archive/path safety helpers."""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

import pytest

from genpy.data.exceptions import UnsafePathError
from genpy.data.paths import (
    ensure_within_root,
    iter_safe_files,
    is_safe_symlink,
    resolve_safe_extraction_path,
    validate_relative_member_path,
)

_UNSAFE_MEMBER_PATHS = (
    "../outside.py",
    "../../secret",
    "/absolute/path.py",
    "C:\\Windows\\system.ini",
    "C:/Windows/system.ini",
    "folder/../../../outside",
    "folder\\..\\..\\outside",
    "",
    "   ",
)


@pytest.mark.parametrize("member", _UNSAFE_MEMBER_PATHS)
def test_unsafe_member_paths_are_rejected(member: str) -> None:
    with pytest.raises(UnsafePathError):
        validate_relative_member_path(member)


def test_safe_member_path_is_accepted() -> None:
    result = validate_relative_member_path("safe/example.py")
    assert result == PurePosixPath("safe/example.py")


def test_null_byte_is_rejected() -> None:
    with pytest.raises(UnsafePathError):
        validate_relative_member_path("safe/example\x00.py")


def test_member_path_normalizing_to_nothing_is_rejected() -> None:
    """A member of "." normalizes to zero path parts and must still be rejected."""
    with pytest.raises(UnsafePathError):
        validate_relative_member_path(".")


def test_resolve_safe_extraction_path_stays_within_root(tmp_path: Path) -> None:
    destination = resolve_safe_extraction_path(tmp_path, "nested/file.py")
    assert destination == (tmp_path / "nested" / "file.py").resolve()


@pytest.mark.parametrize("member", _UNSAFE_MEMBER_PATHS)
def test_resolve_safe_extraction_path_rejects_unsafe_members(tmp_path: Path, member: str) -> None:
    with pytest.raises(UnsafePathError):
        resolve_safe_extraction_path(tmp_path, member)


def test_ensure_within_root_accepts_nested_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    candidate = root / "a" / "b.py"
    result = ensure_within_root(candidate, root)
    assert result == candidate.resolve()


def test_ensure_within_root_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    with pytest.raises(UnsafePathError):
        ensure_within_root(outside, root)


def test_iter_safe_files_lists_all_regular_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.py").write_text("b", encoding="utf-8")

    found = {path.relative_to(tmp_path).as_posix() for path in iter_safe_files(tmp_path)}
    assert found == {"a.py", "nested/b.py"}


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation requires elevated privileges on Windows")
def test_iter_safe_files_skips_unsafe_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside_target = tmp_path / "outside.py"
    outside_target.write_text("secret", encoding="utf-8")

    (root / "inside.py").write_text("inside", encoding="utf-8")
    (root / "escape.py").symlink_to(outside_target)

    found = {path.relative_to(root).as_posix() for path in iter_safe_files(root)}
    assert found == {"inside.py"}


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation requires elevated privileges on Windows")
def test_iter_safe_files_follows_safe_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    real_dir = root / "real"
    real_dir.mkdir()
    (real_dir / "linked.py").write_text("linked", encoding="utf-8")
    (root / "alias").symlink_to(real_dir)

    found = {path.relative_to(root).as_posix() for path in iter_safe_files(root)}
    assert "real/linked.py" in found
    assert "alias/linked.py" in found


def test_is_safe_symlink_resolve_logic_without_real_symlinks(tmp_path: Path) -> None:
    """``is_safe_symlink`` never checks ``is_symlink()`` itself (the caller does),
    so its resolve-and-compare logic can be exercised with plain paths, without
    needing OS-level symlink-creation privileges.
    """
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "nested" / "file.py"
    inside.parent.mkdir()
    inside.write_text("x", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("x", encoding="utf-8")

    assert is_safe_symlink(inside, root) is True
    assert is_safe_symlink(outside, root) is False


def test_is_safe_symlink_returns_false_on_resolve_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    candidate = root / "file.py"

    def _raise_os_error(self: Path, strict: bool = False) -> Path:
        raise OSError("simulated resolve failure")

    monkeypatch.setattr(Path, "resolve", _raise_os_error)
    assert is_safe_symlink(candidate, root) is False


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation requires elevated privileges on Windows")
def test_is_safe_symlink_detects_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("x", encoding="utf-8")
    link = root / "link.py"
    link.symlink_to(outside)

    assert is_safe_symlink(link, root) is False


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation requires elevated privileges on Windows")
def test_is_safe_symlink_accepts_internal_target(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target.py"
    target.write_text("x", encoding="utf-8")
    link = root / "link.py"
    link.symlink_to(target)

    assert is_safe_symlink(link, root) is True
