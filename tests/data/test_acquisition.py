"""Tests for source acquisition: local directory, git repository, and HTTP archive."""

from __future__ import annotations

import io
import stat as stat_module
import tarfile
import zipfile
from email.message import Message
from pathlib import Path
from typing import Any, TypedDict

import pytest

from genpy.data.acquisition import acquire_source, compute_acquisition_identity
from genpy.data.checksums import sha256_file
from genpy.data.exceptions import (
    AcquisitionError,
    ChecksumMismatchError,
    StorageLimitError,
    UnsafePathError,
    UnsupportedSourceTypeError,
)
from genpy.data.manifests import read_manifest
from genpy.data.schemas import AcquisitionSettings, DatasetSource, GovernanceReview, SourceLicense
from tests.data.conftest import LocalGitRepo

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_SOURCE_DIR = FIXTURES_DIR / "sample_source"


class AcquisitionDirs(TypedDict):
    sources_root: Path
    downloads_root: Path
    manifests_dir: Path


def _make_source(**overrides: Any) -> DatasetSource:
    defaults: dict[str, Any] = {
        "id": "acquisition-test",
        "name": "Acquisition Test Source",
        "enabled": True,
        "source_type": "local_directory",
        "location": str(SAMPLE_SOURCE_DIR),
        "description": "Test source.",
        "revision": "fixture-v1",
        "license": SourceLicense(
            declared_spdx="MIT",
            license_file="LICENSE",
            attribution_required=True,
            redistribution_allowed=True,
            commercial_use_allowed=True,
            modifications_allowed=True,
        ),
        "governance": GovernanceReview(
            reviewed_by="Test Suite", reviewed_on="2026-08-01", approval_status="approved"
        ),
        "acquisition": AcquisitionSettings(
            expected_sha256=None,
            maximum_download_bytes=10_485_760,
            maximum_extracted_bytes=10_485_760,
        ),
        "tags": ("python",),
    }
    defaults.update(overrides)
    return DatasetSource(**defaults)


@pytest.fixture
def acquisition_dirs(tmp_path: Path) -> AcquisitionDirs:
    return {
        "sources_root": tmp_path / "raw" / "sources",
        "downloads_root": tmp_path / "raw" / "downloads",
        "manifests_dir": tmp_path / "manifests",
    }


# -- local_directory acquisition --


def test_local_directory_successful_acquisition(acquisition_dirs: AcquisitionDirs) -> None:
    source = _make_source()
    outcome = acquire_source(source, **acquisition_dirs)

    assert outcome.status == "completed"
    assert outcome.destination is not None
    assert (outcome.destination / "example.py").is_file()
    assert (outcome.destination / "LICENSE").is_file()
    assert (outcome.destination / "README.md").is_file()


def test_local_directory_source_remains_unchanged(acquisition_dirs: AcquisitionDirs) -> None:
    original_hash = sha256_file(SAMPLE_SOURCE_DIR / "example.py")
    source = _make_source()
    acquire_source(source, **acquisition_dirs)
    assert sha256_file(SAMPLE_SOURCE_DIR / "example.py") == original_hash


def test_local_directory_generates_manifest(acquisition_dirs: AcquisitionDirs) -> None:
    source = _make_source()
    outcome = acquire_source(source, **acquisition_dirs)

    assert outcome.manifest_path is not None
    manifest_data = read_manifest(outcome.manifest_path)
    assert manifest_data["source_id"] == "acquisition-test"
    assert manifest_data["summary"]["file_count"] == 3


def test_local_directory_manifest_file_hashes_match_disk(acquisition_dirs: AcquisitionDirs) -> None:
    source = _make_source()
    outcome = acquire_source(source, **acquisition_dirs)
    assert outcome.manifest_path is not None
    assert outcome.destination is not None
    manifest_data = read_manifest(outcome.manifest_path)

    for entry in manifest_data["files"]:
        actual = sha256_file(outcome.destination / entry["relative_path"])
        assert actual == entry["sha256"]


def test_local_directory_enforces_byte_limit(acquisition_dirs: AcquisitionDirs) -> None:
    source = _make_source(
        acquisition=AcquisitionSettings(
            expected_sha256=None, maximum_download_bytes=1, maximum_extracted_bytes=1
        )
    )
    with pytest.raises(StorageLimitError):
        acquire_source(source, **acquisition_dirs)

    assert not (acquisition_dirs["sources_root"] / source.id / source.revision).exists()


def test_local_directory_idempotent_rerun_skips(acquisition_dirs: AcquisitionDirs) -> None:
    source = _make_source()
    first = acquire_source(source, **acquisition_dirs)
    second = acquire_source(source, **acquisition_dirs)

    assert first.status == "completed"
    assert second.status == "skipped"


def test_local_directory_force_reacquires(acquisition_dirs: AcquisitionDirs) -> None:
    source = _make_source()
    first = acquire_source(source, **acquisition_dirs)
    assert first.destination is not None

    # Simulate drift: corrupt an acquired file so the manifest no longer matches.
    (first.destination / "example.py").write_text("tampered", encoding="utf-8")

    forced = acquire_source(source, force=True, **acquisition_dirs)
    assert forced.status == "completed"
    assert forced.destination is not None
    assert (forced.destination / "example.py").read_bytes() == (SAMPLE_SOURCE_DIR / "example.py").read_bytes()


def test_local_directory_conflicting_acquisition_requires_force(acquisition_dirs: AcquisitionDirs) -> None:
    source = _make_source()
    first = acquire_source(source, **acquisition_dirs)
    assert first.destination is not None
    (first.destination / "example.py").write_text("tampered", encoding="utf-8")

    with pytest.raises(AcquisitionError, match="does not match its manifest"):
        acquire_source(source, **acquisition_dirs)


def test_local_directory_missing_source_fails(acquisition_dirs: AcquisitionDirs) -> None:
    source = _make_source(location=str(FIXTURES_DIR / "does-not-exist"))
    with pytest.raises(AcquisitionError):
        acquire_source(source, **acquisition_dirs)


def test_dry_run_writes_nothing(acquisition_dirs: AcquisitionDirs) -> None:
    source = _make_source()
    outcome = acquire_source(source, dry_run=True, **acquisition_dirs)

    assert outcome.status == "dry_run"
    assert not acquisition_dirs["sources_root"].exists()
    assert not acquisition_dirs["manifests_dir"].exists()


def test_no_leftover_temp_directory_after_failure(acquisition_dirs: AcquisitionDirs) -> None:
    source = _make_source(
        acquisition=AcquisitionSettings(
            expected_sha256=None, maximum_download_bytes=1, maximum_extracted_bytes=1
        )
    )
    with pytest.raises(StorageLimitError):
        acquire_source(source, **acquisition_dirs)

    parent = acquisition_dirs["sources_root"]
    leftovers = list(parent.glob(".*tmp-*")) if parent.exists() else []
    assert leftovers == []


def test_compute_acquisition_identity_differs_by_revision() -> None:
    source_a = _make_source(revision="v1")
    source_b = _make_source(revision="v2")
    assert compute_acquisition_identity(source_a) != compute_acquisition_identity(source_b)


def test_compute_acquisition_identity_stable_for_same_source() -> None:
    source = _make_source()
    assert compute_acquisition_identity(source) == compute_acquisition_identity(source)


# -- git_repository acquisition --


def test_git_exact_revision_checkout(acquisition_dirs: AcquisitionDirs, local_git_repo: LocalGitRepo) -> None:
    source = _make_source(
        id="git-test",
        source_type="git_repository",
        location=str(local_git_repo.path),
        revision=local_git_repo.first_commit,
    )
    outcome = acquire_source(source, **acquisition_dirs)

    assert outcome.status == "completed"
    assert outcome.destination is not None
    assert (outcome.destination / "first.py").is_file()
    assert not (outcome.destination / "second.py").exists()


def test_git_resolved_commit_is_recorded(
    acquisition_dirs: AcquisitionDirs, local_git_repo: LocalGitRepo
) -> None:
    source = _make_source(
        id="git-test",
        source_type="git_repository",
        location=str(local_git_repo.path),
        revision=local_git_repo.second_commit,
    )
    outcome = acquire_source(source, **acquisition_dirs)
    assert outcome.resolved_revision == local_git_repo.second_commit


def test_git_dot_git_directory_excluded(
    acquisition_dirs: AcquisitionDirs, local_git_repo: LocalGitRepo
) -> None:
    source = _make_source(
        id="git-test",
        source_type="git_repository",
        location=str(local_git_repo.path),
        revision=local_git_repo.second_commit,
    )
    outcome = acquire_source(source, **acquisition_dirs)
    assert outcome.destination is not None
    assert not (outcome.destination / ".git").exists()


def test_git_missing_revision_fails_safely(
    acquisition_dirs: AcquisitionDirs, local_git_repo: LocalGitRepo
) -> None:
    source = _make_source(
        id="git-test",
        source_type="git_repository",
        location=str(local_git_repo.path),
        revision="0000000000000000000000000000000000dead",
    )
    with pytest.raises(AcquisitionError):
        acquire_source(source, **acquisition_dirs)

    assert not (acquisition_dirs["sources_root"] / "git-test" / source.revision).exists()


def test_git_submodules_not_supported(
    acquisition_dirs: AcquisitionDirs, local_git_repo: LocalGitRepo
) -> None:
    source = _make_source(
        id="git-test",
        source_type="git_repository",
        location=str(local_git_repo.path),
        revision=local_git_repo.first_commit,
        acquisition=AcquisitionSettings(
            expected_sha256=None,
            maximum_download_bytes=1_048_576,
            maximum_extracted_bytes=1_048_576,
            include_submodules=True,
        ),
    )
    with pytest.raises(AcquisitionError, match="submodules"):
        acquire_source(source, **acquisition_dirs)


def test_git_temporary_directories_cleaned_up(
    acquisition_dirs: AcquisitionDirs, local_git_repo: LocalGitRepo
) -> None:
    source = _make_source(
        id="git-test",
        source_type="git_repository",
        location=str(local_git_repo.path),
        revision=local_git_repo.first_commit,
    )
    acquire_source(source, **acquisition_dirs)

    parent = acquisition_dirs["sources_root"]
    leftovers = list(parent.glob(".*tmp-*"))
    assert leftovers == []


# -- http_archive acquisition --


class _FakeHTTPResponse:
    """A minimal stand-in for the object returned by ``urllib.request.urlopen``."""

    def __init__(self, data: bytes, headers: dict[str, str] | None = None) -> None:
        self._buffer = io.BytesIO(data)
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value

    def read(self, amount: int = -1) -> bytes:
        return self._buffer.read(amount)

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _patch_urlopen(
    monkeypatch: pytest.MonkeyPatch, payload: bytes, *, headers: dict[str, str] | None = None
) -> None:
    def _fake_urlopen(request: object, timeout: float | None = None) -> _FakeHTTPResponse:  # noqa: ARG001
        return _FakeHTTPResponse(payload, headers)

    monkeypatch.setattr("genpy.data.acquisition.urllib.request.urlopen", _fake_urlopen)


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _zip_bytes_with_symlink(link_name: str, target: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        info = zipfile.ZipInfo(link_name)
        info.external_attr = (stat_module.S_IFLNK | 0o777) << 16
        archive.writestr(info, target)
    return buffer.getvalue()


def _tar_gz_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _tar_gz_bytes_with_symlink(link_name: str, target: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(name=link_name)
        info.type = tarfile.SYMTYPE
        info.linkname = target
        archive.addfile(info)
    return buffer.getvalue()


def _http_source(**overrides: Any) -> DatasetSource:
    defaults: dict[str, Any] = {
        "id": "archive-test",
        "source_type": "http_archive",
        "location": "https://example.invalid/archive.zip",
        "revision": "v1.0.0",
        "acquisition": AcquisitionSettings(
            expected_sha256=None, maximum_download_bytes=1_048_576, maximum_extracted_bytes=1_048_576
        ),
    }
    defaults.update(overrides)
    return _make_source(**defaults)


def test_safe_zip_extraction(monkeypatch: pytest.MonkeyPatch, acquisition_dirs: AcquisitionDirs) -> None:
    payload = _zip_bytes({"a.py": b"print('a')", "nested/b.py": b"print('b')"})
    _patch_urlopen(monkeypatch, payload)

    outcome = acquire_source(_http_source(), **acquisition_dirs)

    assert outcome.status == "completed"
    assert outcome.destination is not None
    assert (outcome.destination / "a.py").read_bytes() == b"print('a')"
    assert (outcome.destination / "nested" / "b.py").read_bytes() == b"print('b')"


def test_safe_tar_extraction(monkeypatch: pytest.MonkeyPatch, acquisition_dirs: AcquisitionDirs) -> None:
    payload = _tar_gz_bytes({"a.py": b"print('a')"})
    _patch_urlopen(monkeypatch, payload)

    source = _http_source(id="tar-test", location="https://example.invalid/archive.tar.gz")
    outcome = acquire_source(source, **acquisition_dirs)

    assert outcome.status == "completed"
    assert outcome.destination is not None
    assert (outcome.destination / "a.py").read_bytes() == b"print('a')"


def test_zip_slip_is_rejected(monkeypatch: pytest.MonkeyPatch, acquisition_dirs: AcquisitionDirs) -> None:
    payload = _zip_bytes({"../evil.py": b"evil"})
    _patch_urlopen(monkeypatch, payload)

    source = _http_source()
    with pytest.raises(UnsafePathError):
        acquire_source(source, **acquisition_dirs)

    assert not (acquisition_dirs["sources_root"] / "archive-test" / source.revision).exists()


def test_tar_traversal_is_rejected(monkeypatch: pytest.MonkeyPatch, acquisition_dirs: AcquisitionDirs) -> None:
    payload = _tar_gz_bytes({"../../evil.py": b"evil"})
    _patch_urlopen(monkeypatch, payload)

    source = _http_source(id="tar-traversal", location="https://example.invalid/archive.tar.gz")
    with pytest.raises(UnsafePathError):
        acquire_source(source, **acquisition_dirs)


def test_zip_symlink_entry_is_rejected(
    monkeypatch: pytest.MonkeyPatch, acquisition_dirs: AcquisitionDirs
) -> None:
    payload = _zip_bytes_with_symlink("link.py", "/etc/passwd")
    _patch_urlopen(monkeypatch, payload)

    with pytest.raises(UnsafePathError):
        acquire_source(_http_source(id="zip-symlink"), **acquisition_dirs)


def test_tar_symlink_entry_is_rejected(
    monkeypatch: pytest.MonkeyPatch, acquisition_dirs: AcquisitionDirs
) -> None:
    payload = _tar_gz_bytes_with_symlink("link.py", "/etc/passwd")
    _patch_urlopen(monkeypatch, payload)

    source = _http_source(id="tar-symlink", location="https://example.invalid/archive.tar.gz")
    with pytest.raises(UnsafePathError):
        acquire_source(source, **acquisition_dirs)


def test_checksum_mismatch_fails_safely(
    monkeypatch: pytest.MonkeyPatch, acquisition_dirs: AcquisitionDirs
) -> None:
    payload = _zip_bytes({"a.py": b"print('a')"})
    _patch_urlopen(monkeypatch, payload)

    source = _http_source(
        acquisition=AcquisitionSettings(
            expected_sha256="0" * 64, maximum_download_bytes=1_048_576, maximum_extracted_bytes=1_048_576
        )
    )
    with pytest.raises(ChecksumMismatchError):
        acquire_source(source, **acquisition_dirs)

    assert list(acquisition_dirs["downloads_root"].glob("*.partial")) == []


def test_download_size_limit_enforced(
    monkeypatch: pytest.MonkeyPatch, acquisition_dirs: AcquisitionDirs
) -> None:
    payload = _zip_bytes({"a.py": b"x" * 10_000})
    _patch_urlopen(monkeypatch, payload)

    source = _http_source(
        acquisition=AcquisitionSettings(
            expected_sha256=None, maximum_download_bytes=100, maximum_extracted_bytes=1_048_576
        )
    )
    with pytest.raises(StorageLimitError):
        acquire_source(source, **acquisition_dirs)

    assert list(acquisition_dirs["downloads_root"].glob("*.partial")) == []


def test_extracted_size_limit_enforced(
    monkeypatch: pytest.MonkeyPatch, acquisition_dirs: AcquisitionDirs
) -> None:
    payload = _zip_bytes({"a.py": b"x" * 100})
    _patch_urlopen(monkeypatch, payload)

    source = _http_source(
        acquisition=AcquisitionSettings(
            expected_sha256=None, maximum_download_bytes=1_048_576, maximum_extracted_bytes=10
        )
    )
    with pytest.raises(StorageLimitError):
        acquire_source(source, **acquisition_dirs)


def test_unsupported_archive_type_is_rejected(acquisition_dirs: AcquisitionDirs) -> None:
    source = _http_source(id="bad-ext", location="https://example.invalid/archive.rar")
    with pytest.raises(UnsupportedSourceTypeError, match="Unsupported archive type"):
        acquire_source(source, **acquisition_dirs)


def test_http_archive_requires_https_at_acquisition_time(acquisition_dirs: AcquisitionDirs) -> None:
    source = _http_source(location="http://example.invalid/archive.zip")
    with pytest.raises(AcquisitionError, match="https"):
        acquire_source(source, **acquisition_dirs)
