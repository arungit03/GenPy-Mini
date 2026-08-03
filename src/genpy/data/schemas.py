"""Typed, immutable dataclasses for the Phase 2 dataset source registry.

This module defines the data model only: field types and single-object
("does this one record make sense on its own?") validation, raised via
``__post_init__``. File-level concerns -- YAML parsing, unknown-field
rejection, duplicate-ID detection, and registry-defaults fallback -- live in
``src/genpy/data/source_registry.py``, which constructs these objects.

The license-policy dataclass lives in ``licenses.py``, manifest dataclasses
in ``manifests.py``, and the report dataclass in ``reporting.py`` -- each
alongside the loader/writer code that owns it.
"""

from __future__ import annotations

import dataclasses
import re
from datetime import date
from typing import Final

from genpy.data.exceptions import SourceRegistryError

SUPPORTED_SOURCE_TYPES: Final[tuple[str, ...]] = (
    "local_directory",
    "git_repository",
    "http_archive",
)

_VALID_APPROVAL_STATUSES: Final[tuple[str, ...]] = (
    "approved",
    "review_required",
    "rejected",
)

_SOURCE_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


@dataclasses.dataclass(frozen=True, slots=True)
class SourceLicense:
    """Human-reviewed licensing metadata for one registered source."""

    declared_spdx: str
    license_file: str | None
    attribution_required: bool
    redistribution_allowed: bool
    commercial_use_allowed: bool
    modifications_allowed: bool
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.declared_spdx.strip():
            raise SourceRegistryError("license.declared_spdx must not be empty.")
        if self.license_file is not None and not self.license_file.strip():
            raise SourceRegistryError("license.license_file must not be blank when provided.")


@dataclasses.dataclass(frozen=True, slots=True)
class GovernanceReview:
    """Human governance sign-off for one registered source."""

    reviewed_by: str
    reviewed_on: str
    approval_status: str
    approval_notes: str = ""

    def __post_init__(self) -> None:
        if not self.reviewed_by.strip():
            raise SourceRegistryError("governance.reviewed_by must not be empty.")

        try:
            date.fromisoformat(self.reviewed_on)
        except ValueError as exc:
            raise SourceRegistryError(
                f"governance.reviewed_on must be an ISO 8601 date (YYYY-MM-DD), "
                f"got {self.reviewed_on!r}."
            ) from exc

        if self.approval_status not in _VALID_APPROVAL_STATUSES:
            raise SourceRegistryError(
                f"governance.approval_status must be one of {_VALID_APPROVAL_STATUSES}, "
                f"got {self.approval_status!r}."
            )


@dataclasses.dataclass(frozen=True, slots=True)
class AcquisitionSettings:
    """Per-source acquisition limits and options."""

    expected_sha256: str | None
    maximum_download_bytes: int
    maximum_extracted_bytes: int
    include_submodules: bool = False
    shallow_clone: bool = True

    def __post_init__(self) -> None:
        if self.maximum_download_bytes <= 0:
            raise SourceRegistryError(
                f"acquisition.maximum_download_bytes must be positive, "
                f"got {self.maximum_download_bytes}."
            )
        if self.maximum_extracted_bytes <= 0:
            raise SourceRegistryError(
                f"acquisition.maximum_extracted_bytes must be positive, "
                f"got {self.maximum_extracted_bytes}."
            )
        if self.expected_sha256 is not None:
            normalized = self.expected_sha256.strip().lower()
            if len(normalized) != 64 or not re.fullmatch(r"[0-9a-f]{64}", normalized):
                raise SourceRegistryError(
                    "acquisition.expected_sha256 must be a 64-character lowercase hex string, "
                    f"got {self.expected_sha256!r}."
                )


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetSource:
    """One registered, reviewable dataset source."""

    id: str
    name: str
    enabled: bool
    source_type: str
    location: str
    description: str
    revision: str
    license: SourceLicense
    governance: GovernanceReview
    acquisition: AcquisitionSettings
    tags: tuple[str, ...] = ()
    homepage: str | None = None
    source_repository: str | None = None
    dataset_card: str | None = None
    citation: str | None = None
    contact: str | None = None
    publication: str | None = None
    known_restrictions: str | None = None
    attribution_text: str | None = None

    def __post_init__(self) -> None:
        if not _SOURCE_ID_RE.fullmatch(self.id):
            raise SourceRegistryError(
                f"Source id {self.id!r} must use only lowercase letters, digits, and hyphens, "
                "and must not start or end with a hyphen."
            )
        if not self.name.strip():
            raise SourceRegistryError(f"Source {self.id!r}: name must not be empty.")
        if self.source_type not in SUPPORTED_SOURCE_TYPES:
            raise SourceRegistryError(
                f"Source {self.id!r}: source_type must be one of {SUPPORTED_SOURCE_TYPES}, "
                f"got {self.source_type!r}."
            )
        if not self.location.strip():
            raise SourceRegistryError(f"Source {self.id!r}: location must not be empty.")
        if not self.revision.strip():
            raise SourceRegistryError(f"Source {self.id!r}: revision must not be empty.")

        normalized_tags = tuple(tag.strip().lower() for tag in self.tags)
        if any(not tag for tag in normalized_tags):
            raise SourceRegistryError(f"Source {self.id!r}: tags must not be blank.")
        if len(set(normalized_tags)) != len(normalized_tags):
            raise SourceRegistryError(f"Source {self.id!r}: tags must be unique, got {self.tags!r}.")
        if normalized_tags != self.tags:
            object.__setattr__(self, "tags", normalized_tags)


@dataclasses.dataclass(frozen=True, slots=True)
class RegistryDefaults:
    """Registry-wide defaults applied when a source omits a setting."""

    enabled: bool
    timeout_seconds: int
    retry_count: int
    maximum_download_bytes: int
    maximum_extracted_bytes: int
    require_pinned_revision: bool
    require_license_metadata: bool
    require_checksum_for_http_archives: bool


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetSourceRegistry:
    """The fully parsed contents of ``config/dataset_sources.yaml``."""

    schema_version: int
    defaults: RegistryDefaults
    sources: tuple[DatasetSource, ...] = ()

    def get(self, source_id: str) -> DatasetSource | None:
        """Return the source with the given id, or ``None`` if not registered."""
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def ids(self) -> tuple[str, ...]:
        """Return every registered source id, in registry order."""
        return tuple(source.id for source in self.sources)
