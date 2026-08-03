"""Tests for provenance manifest construction, writing, and verification."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from genpy.data.checksums import sha256_file
from genpy.data.exceptions import AcquisitionError, DatasetGovernanceError
from genpy.data.manifests import (
    AcquisitionRecord,
    FileManifestRecord,
    build_manifest,
    manifest_path_for,
    manifest_to_dict,
    read_manifest,
    redact_url_credentials,
    sanitize_revision_for_filesystem,
    verify_manifest_against_directory,
    write_manifest,
)
from genpy.data.schemas import DatasetSource
from genpy.data.source_registry import load_source_registry
from tests.data.conftest import base_registry_dict, base_source_dict, write_yaml


def _acquisition_record(**overrides: object) -> AcquisitionRecord:
    now = datetime.now(timezone.utc)
    defaults: dict[str, object] = {
        "started_at": now,
        "completed_at": now,
        "tool_version": "0.1.0",
        "forced": False,
        "governance_override": False,
        "override_reason": None,
    }
    defaults.update(overrides)
    return AcquisitionRecord(**defaults)  # type: ignore[arg-type]


def _load_dataset_source(tmp_path: Path, **overrides: object) -> DatasetSource:
    source_dict = base_source_dict(**overrides)
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([source_dict]))
    return load_source_registry(path).sources[0]


# -- redact_url_credentials --


def test_redact_strips_userinfo() -> None:
    result = redact_url_credentials("https://user:secret@example.invalid/repo.git")
    assert "secret" not in result
    assert "user" not in result
    assert result == "https://example.invalid/repo.git"


def test_redact_strips_query_string() -> None:
    result = redact_url_credentials("https://example.invalid/archive.zip?token=abc123")
    assert "token" not in result
    assert "abc123" not in result


def test_redact_leaves_local_path_unchanged() -> None:
    local_path = "C:/Users/example/data"
    assert redact_url_credentials(local_path) == local_path


# -- AcquisitionRecord --


def test_acquisition_record_requires_timezone_aware_timestamps() -> None:
    with pytest.raises(AcquisitionError):
        AcquisitionRecord(
            started_at=datetime.now(),  # naive
            completed_at=datetime.now(timezone.utc),
            tool_version="0.1.0",
        )


def test_acquisition_record_requires_reason_when_override_true() -> None:
    with pytest.raises(AcquisitionError):
        _acquisition_record(governance_override=True, override_reason="")


def test_acquisition_record_accepts_override_with_reason() -> None:
    record = _acquisition_record(governance_override=True, override_reason="reviewed manually")
    assert record.override_reason == "reviewed manually"


# -- build_manifest / manifest_to_dict --


def test_build_manifest_sorts_files_deterministically(tmp_path: Path) -> None:
    source = _load_dataset_source(tmp_path)
    files = [
        FileManifestRecord(relative_path="b.py", size_bytes=1, sha256="b" * 64),
        FileManifestRecord(relative_path="a.py", size_bytes=2, sha256="a" * 64),
    ]
    manifest = build_manifest(source, "fixture-v1", files, _acquisition_record())
    assert [f.relative_path for f in manifest.files] == ["a.py", "b.py"]


def test_manifest_to_dict_has_expected_shape(tmp_path: Path) -> None:
    source = _load_dataset_source(tmp_path)
    files = [FileManifestRecord(relative_path="example.py", size_bytes=42, sha256="c" * 64)]
    manifest = build_manifest(source, "fixture-v1", files, _acquisition_record())
    payload = manifest_to_dict(manifest)

    assert payload["schema_version"] == 1
    assert payload["source_id"] == source.id
    assert payload["resolved_revision"] == "fixture-v1"
    assert payload["license"]["declared_spdx"] == source.license.declared_spdx
    assert payload["summary"] == {"file_count": 1, "total_bytes": 42}
    assert payload["files"] == [{"relative_path": "example.py", "size_bytes": 42, "sha256": "c" * 64}]
    assert "manifest_digest" in payload
    assert payload["acquisition"]["started_at"].endswith("Z")


def test_manifest_to_dict_redacts_configured_location(tmp_path: Path) -> None:
    source = _load_dataset_source(
        tmp_path,
        source_type="git_repository",
        location="https://user:secret@example.invalid/repo.git",  # pragma: allowlist secret
        revision="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    )
    manifest = build_manifest(source, "deadbeef", [], _acquisition_record())
    payload = manifest_to_dict(manifest)
    assert "secret" not in payload["configured_location"]


def test_manifest_omits_override_reason_when_absent(tmp_path: Path) -> None:
    source = _load_dataset_source(tmp_path)
    manifest = build_manifest(source, "fixture-v1", [], _acquisition_record())
    payload = manifest_to_dict(manifest)
    assert "override_reason" not in payload["acquisition"]


# -- write / read / verify --


def test_write_manifest_produces_valid_utf8_json(tmp_path: Path) -> None:
    source = _load_dataset_source(tmp_path)
    manifest = build_manifest(source, "fixture-v1", [], _acquisition_record())
    manifests_dir = tmp_path / "manifests"
    path = write_manifest(manifest, manifests_dir)

    assert path == manifest_path_for(manifests_dir, source.id, "fixture-v1")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["source_id"] == source.id


def test_sanitize_revision_replaces_path_separators() -> None:
    assert sanitize_revision_for_filesystem("refs/tags/v1.0") == "refs_tags_v1.0"
    assert sanitize_revision_for_filesystem("a\\b/c") == "a_b_c"


def test_read_manifest_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetGovernanceError):
        read_manifest(tmp_path / "missing.json")


def test_read_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DatasetGovernanceError):
        read_manifest(path)


def test_verify_manifest_against_directory_succeeds_when_matching(tmp_path: Path) -> None:
    source_dir = tmp_path / "acquired"
    source_dir.mkdir()
    (source_dir / "example.py").write_bytes(b"content")

    files = [
        FileManifestRecord(
            relative_path="example.py",
            size_bytes=(source_dir / "example.py").stat().st_size,
            sha256=sha256_file(source_dir / "example.py"),
        )
    ]
    manifest_data = {"files": [{"relative_path": f.relative_path, "size_bytes": f.size_bytes, "sha256": f.sha256} for f in files]}

    ok, problems = verify_manifest_against_directory(manifest_data, source_dir)
    assert ok is True
    assert problems == ()


def test_verify_manifest_detects_missing_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "acquired"
    source_dir.mkdir()
    manifest_data = {"files": [{"relative_path": "missing.py", "size_bytes": 1, "sha256": "a" * 64}]}
    ok, problems = verify_manifest_against_directory(manifest_data, source_dir)
    assert ok is False
    assert any("missing file" in problem for problem in problems)


def test_verify_manifest_detects_content_change(tmp_path: Path) -> None:
    source_dir = tmp_path / "acquired"
    source_dir.mkdir()
    target = source_dir / "example.py"
    target.write_bytes(b"original")
    original_sha = sha256_file(target)

    target.write_bytes(b"tampered!!")  # same-ish length changes possible; use distinct sha check
    manifest_data = {
        "files": [{"relative_path": "example.py", "size_bytes": len(b"original"), "sha256": original_sha}]
    }
    ok, problems = verify_manifest_against_directory(manifest_data, source_dir)
    assert ok is False


def test_verify_manifest_detects_unexpected_extra_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "acquired"
    source_dir.mkdir()
    (source_dir / "extra.py").write_bytes(b"x")
    ok, problems = verify_manifest_against_directory({"files": []}, source_dir)
    assert ok is False
    assert any("extra.py" in problem for problem in problems)
