"""Tests for dataset source registry loading, validation, and governance evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from genpy.data.exceptions import SourceRegistryError
from genpy.data.licenses import load_license_policy
from genpy.data.source_registry import evaluate_registry, evaluate_source, load_source_registry
from tests.data.conftest import (
    PolicyPathFactory,
    base_policy_dict,
    base_registry_dict,
    base_source_dict,
    write_yaml,
)


def test_valid_empty_registry_loads(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict())
    registry = load_source_registry(path)
    assert registry.schema_version == 1
    assert registry.sources == ()


def test_valid_local_source_loads(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([base_source_dict()]))
    registry = load_source_registry(path)
    assert registry.ids() == ("sample-source",)


def test_valid_git_source_loads(tmp_path: Path) -> None:
    source = base_source_dict(
        id="git-source",
        source_type="git_repository",
        location="https://example.invalid/repo.git",
        revision="abc123def456abc123def456abc123def456abc",  # pragma: allowlist secret
    )
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([source]))
    registry = load_source_registry(path)
    assert registry.get("git-source") is not None


def test_valid_http_source_loads(tmp_path: Path) -> None:
    source = base_source_dict(
        id="http-source",
        source_type="http_archive",
        location="https://example.invalid/archive.zip",
        revision="v1.0.0",
        acquisition={
            "expected_sha256": "a" * 64,
            "maximum_download_bytes": 1024,
            "maximum_extracted_bytes": 1024,
            "include_submodules": False,
            "shallow_clone": True,
        },
    )
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([source]))
    registry = load_source_registry(path)
    assert registry.get("http-source") is not None


def test_duplicate_source_ids_are_rejected(tmp_path: Path) -> None:
    sources = [base_source_dict(id="dup"), base_source_dict(id="dup", name="Second")]
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict(sources))
    with pytest.raises(SourceRegistryError, match="duplicate"):
        load_source_registry(path)


def test_invalid_source_id_is_rejected(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "registry.yaml", base_registry_dict([base_source_dict(id="Bad ID!")])
    )
    with pytest.raises(SourceRegistryError):
        load_source_registry(path)


def test_missing_revision_is_rejected(tmp_path: Path) -> None:
    source = base_source_dict()
    del source["revision"]
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([source]))
    with pytest.raises(SourceRegistryError, match="revision"):
        load_source_registry(path)


def test_unsupported_source_type_is_rejected(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "registry.yaml", base_registry_dict([base_source_dict(source_type="ftp_mirror")])
    )
    with pytest.raises(SourceRegistryError):
        load_source_registry(path)


def test_invalid_approval_status_is_rejected(tmp_path: Path) -> None:
    source = base_source_dict(
        governance={
            "reviewed_by": "Reviewer",
            "reviewed_on": "2026-08-01",
            "approval_status": "maybe",
            "approval_notes": "",
        }
    )
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([source]))
    with pytest.raises(SourceRegistryError):
        load_source_registry(path)


def test_missing_license_information_is_rejected(tmp_path: Path) -> None:
    source = base_source_dict()
    del source["license"]["declared_spdx"]
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([source]))
    with pytest.raises(SourceRegistryError, match="declared_spdx"):
        load_source_registry(path)


def test_invalid_review_date_is_rejected(tmp_path: Path) -> None:
    source = base_source_dict(
        governance={
            "reviewed_by": "Reviewer",
            "reviewed_on": "not-a-date",
            "approval_status": "approved",
            "approval_notes": "",
        }
    )
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([source]))
    with pytest.raises(SourceRegistryError):
        load_source_registry(path)


def test_invalid_byte_limits_are_rejected(tmp_path: Path) -> None:
    source = base_source_dict(
        acquisition={
            "expected_sha256": None,
            "maximum_download_bytes": 0,
            "maximum_extracted_bytes": 1024,
            "include_submodules": False,
            "shallow_clone": True,
        }
    )
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([source]))
    with pytest.raises(SourceRegistryError):
        load_source_registry(path)


def test_duplicate_tags_are_rejected(tmp_path: Path) -> None:
    source = base_source_dict(tags=["python", "Python"])
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([source]))
    with pytest.raises(SourceRegistryError):
        load_source_registry(path)


def test_unknown_critical_field_on_source_is_rejected(tmp_path: Path) -> None:
    source = base_source_dict()
    source["locaton"] = source["location"]  # typo'd critical field, alongside the correct one
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([source]))
    with pytest.raises(SourceRegistryError, match="unknown field"):
        load_source_registry(path)


def test_unknown_field_on_license_block_is_rejected(tmp_path: Path) -> None:
    source = base_source_dict()
    source["license"]["extra_bogus_field"] = True
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([source]))
    with pytest.raises(SourceRegistryError, match="unknown field"):
        load_source_registry(path)


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    path.write_text("sources: [unterminated", encoding="utf-8")
    with pytest.raises(SourceRegistryError, match="YAML"):
        load_source_registry(path)


def test_missing_registry_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SourceRegistryError, match="not found"):
        load_source_registry(tmp_path / "missing.yaml")


def test_git_source_requires_pinned_revision_when_default_requires_it(tmp_path: Path) -> None:
    source = base_source_dict(
        id="floating",
        source_type="git_repository",
        location="https://example.invalid/repo.git",
        revision="main",
    )
    path = write_yaml(
        tmp_path / "registry.yaml", base_registry_dict([source], require_pinned_revision=True)
    )
    with pytest.raises(SourceRegistryError, match="pinned"):
        load_source_registry(path)


def test_git_source_floating_ref_allowed_when_not_required(tmp_path: Path) -> None:
    source = base_source_dict(
        id="floating",
        source_type="git_repository",
        location="https://example.invalid/repo.git",
        revision="main",
    )
    path = write_yaml(
        tmp_path / "registry.yaml", base_registry_dict([source], require_pinned_revision=False)
    )
    registry = load_source_registry(path)
    assert registry.get("floating") is not None


def test_http_archive_requires_checksum_when_default_requires_it(tmp_path: Path) -> None:
    source = base_source_dict(
        id="http-source",
        source_type="http_archive",
        location="https://example.invalid/archive.zip",
    )
    path = write_yaml(
        tmp_path / "registry.yaml",
        base_registry_dict([source], require_checksum_for_http_archives=True),
    )
    with pytest.raises(SourceRegistryError, match="expected_sha256"):
        load_source_registry(path)


def test_http_archive_requires_https(tmp_path: Path) -> None:
    source = base_source_dict(
        id="http-source",
        source_type="http_archive",
        location="http://example.invalid/archive.zip",
        acquisition={
            "expected_sha256": "a" * 64,
            "maximum_download_bytes": 1024,
            "maximum_extracted_bytes": 1024,
            "include_submodules": False,
            "shallow_clone": True,
        },
    )
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([source]))
    with pytest.raises(SourceRegistryError, match="https"):
        load_source_registry(path)


def test_http_archive_allows_localhost_for_testing(tmp_path: Path) -> None:
    source = base_source_dict(
        id="http-source",
        source_type="http_archive",
        location="http://localhost:8000/archive.zip",
        acquisition={
            "expected_sha256": "a" * 64,
            "maximum_download_bytes": 1024,
            "maximum_extracted_bytes": 1024,
            "include_submodules": False,
            "shallow_clone": True,
        },
    )
    path = write_yaml(tmp_path / "registry.yaml", base_registry_dict([source]))
    registry = load_source_registry(path)
    assert registry.get("http-source") is not None


def test_requires_license_file_when_default_requires_it(tmp_path: Path) -> None:
    source = base_source_dict()
    del source["license"]["license_file"]
    path = write_yaml(
        tmp_path / "registry.yaml", base_registry_dict([source], require_license_metadata=True)
    )
    with pytest.raises(SourceRegistryError, match="license_file"):
        load_source_registry(path)


# -- governance evaluation --


def test_evaluate_source_approved(tmp_path: Path, policy_path_factory: PolicyPathFactory) -> None:
    policy = load_license_policy(policy_path_factory())
    source = load_source_registry(
        write_yaml(tmp_path / "registry.yaml", base_registry_dict([base_source_dict()]))
    ).sources[0]
    evaluation = evaluate_source(source, policy)
    assert evaluation.effective_status == "approved"
    assert evaluation.reasons == ()


def test_evaluate_source_rejected_by_governance(
    tmp_path: Path, policy_path_factory: PolicyPathFactory
) -> None:
    policy = load_license_policy(policy_path_factory())
    source_dict = base_source_dict(
        governance={
            "reviewed_by": "Reviewer",
            "reviewed_on": "2026-08-01",
            "approval_status": "rejected",
            "approval_notes": "no",
        }
    )
    source = load_source_registry(
        write_yaml(tmp_path / "registry.yaml", base_registry_dict([source_dict]))
    ).sources[0]
    evaluation = evaluate_source(source, policy)
    assert evaluation.effective_status == "rejected"


def test_evaluate_source_rejected_by_blocked_license(
    tmp_path: Path, policy_path_factory: PolicyPathFactory
) -> None:
    policy = load_license_policy(policy_path_factory())
    source_dict = base_source_dict(
        license={
            "declared_spdx": "GPL-3.0-only",
            "license_file": "LICENSE",
            "attribution_required": True,
            "redistribution_allowed": False,
            "commercial_use_allowed": False,
            "modifications_allowed": True,
            "notes": "",
        }
    )
    source = load_source_registry(
        write_yaml(tmp_path / "registry.yaml", base_registry_dict([source_dict]))
    ).sources[0]
    evaluation = evaluate_source(source, policy)
    assert evaluation.effective_status == "rejected"
    assert evaluation.license_status == "blocked"


def test_evaluate_source_review_required_by_governance(
    tmp_path: Path, policy_path_factory: PolicyPathFactory
) -> None:
    policy = load_license_policy(policy_path_factory())
    source_dict = base_source_dict(
        governance={
            "reviewed_by": "Reviewer",
            "reviewed_on": "2026-08-01",
            "approval_status": "review_required",
            "approval_notes": "",
        }
    )
    source = load_source_registry(
        write_yaml(tmp_path / "registry.yaml", base_registry_dict([source_dict]))
    ).sources[0]
    evaluation = evaluate_source(source, policy)
    assert evaluation.effective_status == "review_required"


def test_evaluate_registry_returns_all_sources_in_order(
    tmp_path: Path, policy_path_factory: PolicyPathFactory
) -> None:
    policy = load_license_policy(policy_path_factory())
    sources = [base_source_dict(id="first"), base_source_dict(id="second")]
    registry = load_source_registry(
        write_yaml(tmp_path / "registry.yaml", base_registry_dict(sources))
    )
    evaluations = evaluate_registry(registry, policy)
    assert [e.source.id for e in evaluations] == ["first", "second"]
