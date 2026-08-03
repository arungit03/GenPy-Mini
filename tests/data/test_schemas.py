"""Tests for the Phase 2 dataset source dataclasses and their self-validation."""

from __future__ import annotations

import dataclasses

import pytest

from genpy.data.exceptions import SourceRegistryError
from genpy.data.schemas import (
    AcquisitionSettings,
    DatasetSource,
    GovernanceReview,
    SourceLicense,
)


def _license(**overrides: object) -> SourceLicense:
    defaults: dict[str, object] = {
        "declared_spdx": "MIT",
        "license_file": "LICENSE",
        "attribution_required": True,
        "redistribution_allowed": True,
        "commercial_use_allowed": True,
        "modifications_allowed": True,
        "notes": "",
    }
    defaults.update(overrides)
    return SourceLicense(**defaults)  # type: ignore[arg-type]


def _governance(**overrides: object) -> GovernanceReview:
    defaults: dict[str, object] = {
        "reviewed_by": "Reviewer",
        "reviewed_on": "2026-08-01",
        "approval_status": "approved",
        "approval_notes": "",
    }
    defaults.update(overrides)
    return GovernanceReview(**defaults)  # type: ignore[arg-type]


def _acquisition(**overrides: object) -> AcquisitionSettings:
    defaults: dict[str, object] = {
        "expected_sha256": None,
        "maximum_download_bytes": 1024,
        "maximum_extracted_bytes": 1024,
        "include_submodules": False,
        "shallow_clone": True,
    }
    defaults.update(overrides)
    return AcquisitionSettings(**defaults)  # type: ignore[arg-type]


def _source(**overrides: object) -> DatasetSource:
    defaults: dict[str, object] = {
        "id": "sample-id",
        "name": "Sample",
        "enabled": True,
        "source_type": "local_directory",
        "location": "/tmp/sample",
        "description": "A sample.",
        "revision": "v1",
        "license": _license(),
        "governance": _governance(),
        "acquisition": _acquisition(),
        "tags": ("python",),
    }
    defaults.update(overrides)
    return DatasetSource(**defaults)  # type: ignore[arg-type]


# -- SourceLicense --


def test_license_requires_non_empty_spdx() -> None:
    with pytest.raises(SourceRegistryError):
        _license(declared_spdx="  ")


def test_license_rejects_blank_license_file() -> None:
    with pytest.raises(SourceRegistryError):
        _license(license_file="   ")


def test_license_allows_missing_license_file() -> None:
    license_ = _license(license_file=None)
    assert license_.license_file is None


# -- GovernanceReview --


def test_governance_requires_valid_iso_date() -> None:
    with pytest.raises(SourceRegistryError):
        _governance(reviewed_on="08/01/2026")


def test_governance_rejects_invalid_approval_status() -> None:
    with pytest.raises(SourceRegistryError):
        _governance(approval_status="maybe")


@pytest.mark.parametrize("status", ["approved", "review_required", "rejected"])
def test_governance_accepts_valid_approval_statuses(status: str) -> None:
    review = _governance(approval_status=status)
    assert review.approval_status == status


def test_governance_requires_non_empty_reviewer() -> None:
    with pytest.raises(SourceRegistryError):
        _governance(reviewed_by=" ")


# -- AcquisitionSettings --


def test_acquisition_rejects_non_positive_download_limit() -> None:
    with pytest.raises(SourceRegistryError):
        _acquisition(maximum_download_bytes=0)


def test_acquisition_rejects_non_positive_extracted_limit() -> None:
    with pytest.raises(SourceRegistryError):
        _acquisition(maximum_extracted_bytes=-1)


def test_acquisition_rejects_malformed_sha256() -> None:
    with pytest.raises(SourceRegistryError):
        _acquisition(expected_sha256="not-a-hash")


def test_acquisition_accepts_valid_sha256() -> None:
    digest = "a" * 64
    settings = _acquisition(expected_sha256=digest)
    assert settings.expected_sha256 == digest


# -- DatasetSource --


@pytest.mark.parametrize("bad_id", ["Bad_ID", "UPPER", "trailing-", "-leading", "has space", "has_underscore"])
def test_source_rejects_invalid_id_formats(bad_id: str) -> None:
    with pytest.raises(SourceRegistryError):
        _source(id=bad_id)


@pytest.mark.parametrize("good_id", ["simple", "with-hyphen", "digits123", "a1-b2-c3"])
def test_source_accepts_valid_id_formats(good_id: str) -> None:
    source = _source(id=good_id)
    assert source.id == good_id


def test_source_rejects_unsupported_source_type() -> None:
    with pytest.raises(SourceRegistryError):
        _source(source_type="ftp_mirror")


def test_source_rejects_empty_location() -> None:
    with pytest.raises(SourceRegistryError):
        _source(location="  ")


def test_source_rejects_empty_revision() -> None:
    with pytest.raises(SourceRegistryError):
        _source(revision="")


def test_source_rejects_empty_name() -> None:
    with pytest.raises(SourceRegistryError):
        _source(name="")


def test_source_rejects_duplicate_tags_case_insensitive() -> None:
    with pytest.raises(SourceRegistryError):
        _source(tags=("Python", "python"))


def test_source_normalizes_tag_casing_and_whitespace() -> None:
    source = _source(tags=(" Python ", "Fixture"))
    assert source.tags == ("python", "fixture")


def test_source_rejects_blank_tag() -> None:
    with pytest.raises(SourceRegistryError):
        _source(tags=("python", "  "))


def test_source_is_immutable() -> None:
    source = _source()
    with pytest.raises(dataclasses.FrozenInstanceError):
        source.name = "changed"  # type: ignore[misc]
