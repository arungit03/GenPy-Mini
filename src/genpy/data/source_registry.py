"""Loading, validation, and governance evaluation for the dataset source registry.

This module turns ``config/dataset_sources.yaml`` into validated
:class:`~genpy.data.schemas.DatasetSourceRegistry` objects, and combines a
registry with a :class:`~genpy.data.licenses.LicensePolicy` to decide
whether each source is ``approved``, ``review_required``, or ``rejected``.

Approval is never inferred from public availability alone: a source is
``approved`` only when both its human governance review says ``approved``
*and* its declared license evaluates to ``allowed`` under the active
policy.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from pathlib import Path
from typing import Any, Final

import yaml

from genpy.data.exceptions import SourceRegistryError
from genpy.data.licenses import LicensePolicy
from genpy.data.schemas import (
    AcquisitionSettings,
    DatasetSource,
    DatasetSourceRegistry,
    GovernanceReview,
    RegistryDefaults,
    SourceLicense,
)

_REQUIRED_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = ("schema_version", "defaults", "sources")
_KNOWN_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(_REQUIRED_TOP_LEVEL_FIELDS)

_REQUIRED_DEFAULTS_FIELDS: Final[tuple[str, ...]] = (
    "enabled",
    "timeout_seconds",
    "retry_count",
    "maximum_download_bytes",
    "maximum_extracted_bytes",
    "require_pinned_revision",
    "require_license_metadata",
    "require_checksum_for_http_archives",
)
_KNOWN_DEFAULTS_FIELDS: Final[frozenset[str]] = frozenset(_REQUIRED_DEFAULTS_FIELDS)

_REQUIRED_SOURCE_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "name",
    "enabled",
    "source_type",
    "location",
    "description",
    "revision",
    "license",
    "governance",
)
_OPTIONAL_SOURCE_FIELDS: Final[tuple[str, ...]] = (
    "acquisition",
    "tags",
    "homepage",
    "source_repository",
    "dataset_card",
    "citation",
    "contact",
    "publication",
    "known_restrictions",
    "attribution_text",
)
_KNOWN_SOURCE_FIELDS: Final[frozenset[str]] = frozenset(_REQUIRED_SOURCE_FIELDS) | frozenset(
    _OPTIONAL_SOURCE_FIELDS
)

_REQUIRED_LICENSE_FIELDS: Final[tuple[str, ...]] = (
    "declared_spdx",
    "attribution_required",
    "redistribution_allowed",
    "commercial_use_allowed",
    "modifications_allowed",
)
_KNOWN_LICENSE_FIELDS: Final[frozenset[str]] = frozenset(_REQUIRED_LICENSE_FIELDS) | frozenset(
    ("license_file", "notes")
)

_REQUIRED_GOVERNANCE_FIELDS: Final[tuple[str, ...]] = ("reviewed_by", "reviewed_on", "approval_status")
_KNOWN_GOVERNANCE_FIELDS: Final[frozenset[str]] = frozenset(_REQUIRED_GOVERNANCE_FIELDS) | frozenset(
    ("approval_notes",)
)

_KNOWN_ACQUISITION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "expected_sha256",
        "maximum_download_bytes",
        "maximum_extracted_bytes",
        "include_submodules",
        "shallow_clone",
    }
)

_FLOATING_GIT_REFS: Final[frozenset[str]] = frozenset({"head", "main", "master", "latest", "trunk"})


def load_source_registry(path: Path) -> DatasetSourceRegistry:
    """Load, validate, and return the dataset source registry at ``path``.

    Raises:
        SourceRegistryError: for a missing file, invalid YAML, a missing or
            unknown field anywhere in the document, a duplicate source id,
            or a failed cross-field consistency check.
    """
    data = _read_yaml_mapping(path)
    _require_fields(data, _REQUIRED_TOP_LEVEL_FIELDS, path, "registry")
    _reject_unknown_fields(data, _KNOWN_TOP_LEVEL_FIELDS, path, "registry")

    try:
        schema_version = int(data["schema_version"])
    except (TypeError, ValueError) as exc:
        raise SourceRegistryError(f"{path}: 'schema_version' must be an integer.") from exc

    defaults = _build_defaults(data["defaults"], path)

    raw_sources = data["sources"]
    if not isinstance(raw_sources, list):
        raise SourceRegistryError(f"{path}: 'sources' must be a list.")

    sources = tuple(_build_source(item, defaults, path) for item in raw_sources)
    _reject_duplicate_ids(sources, path)

    return DatasetSourceRegistry(schema_version=schema_version, defaults=defaults, sources=sources)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SourceRegistryError(f"Dataset source registry file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SourceRegistryError(f"Could not read dataset source registry file {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise SourceRegistryError(f"Dataset source registry file {path} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise SourceRegistryError(
            f"Dataset source registry file {path} must contain a mapping at the top level."
        )

    return data


def _require_fields(data: dict[str, Any], fields: tuple[str, ...], path: Path, context: str) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        raise SourceRegistryError(f"{path}: {context} is missing required field(s): {', '.join(missing)}")


def _reject_unknown_fields(data: dict[str, Any], known: frozenset[str], path: Path, context: str) -> None:
    unknown = sorted(set(data) - known)
    if unknown:
        raise SourceRegistryError(f"{path}: {context} has unknown field(s): {', '.join(unknown)}")


def _reject_duplicate_ids(sources: tuple[DatasetSource, ...], path: Path) -> None:
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.id] = counts.get(source.id, 0) + 1
    duplicates = sorted(source_id for source_id, count in counts.items() if count > 1)
    if duplicates:
        raise SourceRegistryError(f"{path}: duplicate source id(s): {', '.join(duplicates)}")


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _build_defaults(raw: Any, path: Path) -> RegistryDefaults:
    if not isinstance(raw, dict):
        raise SourceRegistryError(f"{path}: 'defaults' must be a mapping.")
    _require_fields(raw, _REQUIRED_DEFAULTS_FIELDS, path, "defaults")
    _reject_unknown_fields(raw, _KNOWN_DEFAULTS_FIELDS, path, "defaults")

    try:
        defaults = RegistryDefaults(
            enabled=bool(raw["enabled"]),
            timeout_seconds=int(raw["timeout_seconds"]),
            retry_count=int(raw["retry_count"]),
            maximum_download_bytes=int(raw["maximum_download_bytes"]),
            maximum_extracted_bytes=int(raw["maximum_extracted_bytes"]),
            require_pinned_revision=bool(raw["require_pinned_revision"]),
            require_license_metadata=bool(raw["require_license_metadata"]),
            require_checksum_for_http_archives=bool(raw["require_checksum_for_http_archives"]),
        )
    except (TypeError, ValueError) as exc:
        raise SourceRegistryError(f"{path}: 'defaults' is malformed: {exc}") from exc

    if defaults.timeout_seconds <= 0:
        raise SourceRegistryError(f"{path}: defaults.timeout_seconds must be positive.")
    if defaults.retry_count < 0:
        raise SourceRegistryError(f"{path}: defaults.retry_count must not be negative.")
    if defaults.maximum_download_bytes <= 0:
        raise SourceRegistryError(f"{path}: defaults.maximum_download_bytes must be positive.")
    if defaults.maximum_extracted_bytes <= 0:
        raise SourceRegistryError(f"{path}: defaults.maximum_extracted_bytes must be positive.")

    return defaults


def _build_license(raw: Any, source_id: Any, path: Path) -> SourceLicense:
    if not isinstance(raw, dict):
        raise SourceRegistryError(f"{path}: source {source_id!r}: 'license' must be a mapping.")
    _require_fields(raw, _REQUIRED_LICENSE_FIELDS, path, f"source {source_id!r} license")
    _reject_unknown_fields(raw, _KNOWN_LICENSE_FIELDS, path, f"source {source_id!r} license")

    return SourceLicense(
        declared_spdx=str(raw["declared_spdx"]),
        license_file=_optional_str(raw.get("license_file")),
        attribution_required=bool(raw["attribution_required"]),
        redistribution_allowed=bool(raw["redistribution_allowed"]),
        commercial_use_allowed=bool(raw["commercial_use_allowed"]),
        modifications_allowed=bool(raw["modifications_allowed"]),
        notes=str(raw.get("notes", "")),
    )


def _build_governance(raw: Any, source_id: Any, path: Path) -> GovernanceReview:
    if not isinstance(raw, dict):
        raise SourceRegistryError(f"{path}: source {source_id!r}: 'governance' must be a mapping.")
    _require_fields(raw, _REQUIRED_GOVERNANCE_FIELDS, path, f"source {source_id!r} governance")
    _reject_unknown_fields(raw, _KNOWN_GOVERNANCE_FIELDS, path, f"source {source_id!r} governance")

    reviewed_on = raw["reviewed_on"]
    reviewed_on_str = reviewed_on.isoformat() if isinstance(reviewed_on, date) else str(reviewed_on)

    return GovernanceReview(
        reviewed_by=str(raw["reviewed_by"]),
        reviewed_on=reviewed_on_str,
        approval_status=str(raw["approval_status"]),
        approval_notes=str(raw.get("approval_notes", "")),
    )


def _build_acquisition(
    raw: Any, defaults: RegistryDefaults, source_id: Any, path: Path
) -> AcquisitionSettings:
    if not isinstance(raw, dict):
        raise SourceRegistryError(f"{path}: source {source_id!r}: 'acquisition' must be a mapping.")
    _reject_unknown_fields(raw, _KNOWN_ACQUISITION_FIELDS, path, f"source {source_id!r} acquisition")

    try:
        return AcquisitionSettings(
            expected_sha256=_optional_str(raw.get("expected_sha256")),
            maximum_download_bytes=int(
                raw.get("maximum_download_bytes", defaults.maximum_download_bytes)
            ),
            maximum_extracted_bytes=int(
                raw.get("maximum_extracted_bytes", defaults.maximum_extracted_bytes)
            ),
            include_submodules=bool(raw.get("include_submodules", False)),
            shallow_clone=bool(raw.get("shallow_clone", True)),
        )
    except (TypeError, ValueError) as exc:
        raise SourceRegistryError(
            f"{path}: source {source_id!r}: 'acquisition' is malformed: {exc}"
        ) from exc


def _build_source(raw: Any, defaults: RegistryDefaults, path: Path) -> DatasetSource:
    if not isinstance(raw, dict):
        raise SourceRegistryError(f"{path}: each entry in 'sources' must be a mapping.")

    source_id = raw.get("id", "<unknown>")
    _require_fields(raw, _REQUIRED_SOURCE_FIELDS, path, f"source {source_id!r}")
    _reject_unknown_fields(raw, _KNOWN_SOURCE_FIELDS, path, f"source {source_id!r}")

    tags_raw = raw.get("tags", [])
    if not isinstance(tags_raw, list):
        raise SourceRegistryError(f"{path}: source {source_id!r}: 'tags' must be a list.")

    source = DatasetSource(
        id=str(raw["id"]),
        name=str(raw["name"]),
        enabled=bool(raw["enabled"]),
        source_type=str(raw["source_type"]),
        location=str(raw["location"]),
        description=str(raw["description"]),
        revision=str(raw["revision"]),
        license=_build_license(raw["license"], source_id, path),
        governance=_build_governance(raw["governance"], source_id, path),
        acquisition=_build_acquisition(raw.get("acquisition", {}), defaults, source_id, path),
        tags=tuple(str(tag) for tag in tags_raw),
        homepage=_optional_str(raw.get("homepage")),
        source_repository=_optional_str(raw.get("source_repository")),
        dataset_card=_optional_str(raw.get("dataset_card")),
        citation=_optional_str(raw.get("citation")),
        contact=_optional_str(raw.get("contact")),
        publication=_optional_str(raw.get("publication")),
        known_restrictions=_optional_str(raw.get("known_restrictions")),
        attribution_text=_optional_str(raw.get("attribution_text")),
    )

    _validate_source_type_specific_rules(source, defaults, path)
    return source


def _is_local_test_host(location: str) -> bool:
    return location.startswith("http://localhost") or location.startswith("http://127.0.0.1")


def _validate_source_type_specific_rules(
    source: DatasetSource, defaults: RegistryDefaults, path: Path
) -> None:
    if defaults.require_license_metadata and not source.license.license_file:
        raise SourceRegistryError(
            f"{path}: source {source.id!r} must set license.license_file "
            "(defaults.require_license_metadata is true)."
        )

    if source.source_type == "git_repository":
        if defaults.require_pinned_revision and source.revision.strip().lower() in _FLOATING_GIT_REFS:
            raise SourceRegistryError(
                f"{path}: source {source.id!r} is a git_repository and "
                "defaults.require_pinned_revision is true, so revision must be an immutable "
                f"commit hash or tag, not a floating ref like {source.revision!r}."
            )
        if source.location.startswith("http://") and not _is_local_test_host(source.location):
            raise SourceRegistryError(
                f"{path}: source {source.id!r} must use https:// for a remote git location, "
                f"got {source.location!r}."
            )

    if source.source_type == "http_archive":
        if source.location.startswith("http://") and not _is_local_test_host(source.location):
            raise SourceRegistryError(
                f"{path}: source {source.id!r} must use https:// unless the location is "
                f"http://localhost or http://127.0.0.1 for local testing, got {source.location!r}."
            )
        if defaults.require_checksum_for_http_archives and not source.acquisition.expected_sha256:
            raise SourceRegistryError(
                f"{path}: source {source.id!r} is an http_archive and "
                "defaults.require_checksum_for_http_archives is true, so "
                "acquisition.expected_sha256 is required."
            )


@dataclasses.dataclass(frozen=True, slots=True)
class SourceEvaluation:
    """The combined governance + license decision for one registered source."""

    source: DatasetSource
    license_status: str
    effective_status: str
    reasons: tuple[str, ...]


def evaluate_source(source: DatasetSource, policy: LicensePolicy) -> SourceEvaluation:
    """Combine governance review and license policy into one conservative decision.

    A source is ``approved`` only when governance review says ``approved``
    *and* the declared license evaluates to ``allowed``. Either dimension
    saying ``rejected``/``blocked`` makes the source ``rejected`` outright;
    otherwise any ``review_required`` signal makes the source
    ``review_required``. Public availability of the source code is never,
    by itself, treated as approval.
    """
    license_status = policy.evaluate(source.license.declared_spdx)
    reasons: list[str] = []

    if source.governance.approval_status == "rejected":
        reasons.append("governance approval_status is 'rejected'")
    if license_status == "blocked":
        reasons.append(f"license {source.license.declared_spdx!r} is blocked by policy")

    if reasons:
        return SourceEvaluation(source, license_status, "rejected", tuple(reasons))

    if source.governance.approval_status == "review_required":
        reasons.append("governance approval_status is 'review_required'")
    if license_status == "review_required":
        reasons.append(f"license {source.license.declared_spdx!r} requires human review")

    if reasons:
        return SourceEvaluation(source, license_status, "review_required", tuple(reasons))

    if source.governance.approval_status == "approved" and license_status == "allowed":
        return SourceEvaluation(source, license_status, "approved", ())

    return SourceEvaluation(
        source,
        license_status,
        "review_required",
        ("governance and license status combination was not confidently approvable",),
    )


def evaluate_registry(
    registry: DatasetSourceRegistry, policy: LicensePolicy
) -> tuple[SourceEvaluation, ...]:
    """Evaluate every source in ``registry`` against ``policy``, in registry order."""
    return tuple(evaluate_source(source, policy) for source in registry.sources)
