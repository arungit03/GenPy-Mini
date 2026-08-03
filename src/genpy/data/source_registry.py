"""Configuration-backed source registry."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SourceEntry:
    """An auditable source definition with pinned provenance."""

    id: str
    name: str
    official_url: str
    dataset_card_url: str | None
    version: str
    access_method: str
    expected_download_size: str
    expected_extracted_size: str
    expected_usable_records: str
    estimated_disk_requirement: str
    streaming_supported: bool
    languages: tuple[str, ...]
    dataset_level_licence: str
    per_record_licence: bool
    provenance_available: bool
    opt_out_supported: bool
    attribution_required: bool
    status: str
    review_notes: str
    archive_url: str | None = None
    checksum_sha256: str | None = None
    repository: str | None = None
    release_tag: str | None = None
    include_globs: tuple[str, ...] = field(default_factory=lambda: ("*.py",))
    local_path: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceEntry:
        """Validate and construct a source entry."""
        data = dict(value)
        data["languages"] = tuple(data.get("languages", ()))
        data["include_globs"] = tuple(data.get("include_globs", ("*.py",)))
        source = cls(**data)
        source.validate()
        return source

    def validate(self) -> None:
        """Require the provenance and access metadata used by the pipeline."""
        fields = {
            "id": self.id,
            "name": self.name,
            "official_url": self.official_url,
            "version": self.version,
            "access_method": self.access_method,
            "status": self.status,
        }
        missing = [name for name, value in fields.items() if not value]
        if missing:
            raise ValueError(f"source is missing fields: {', '.join(missing)}")
        if "Python" not in self.languages:
            raise ValueError(f"source {self.id} is not declared as Python")
        if self.access_method == "github_archive" and not self.archive_url:
            raise ValueError(f"source {self.id} requires archive_url")
        if self.access_method == "local_directory" and not self.local_path:
            raise ValueError(f"source {self.id} requires local_path")


class SourceRegistry:
    """Lazily expose source entries loaded from YAML."""

    def __init__(self, entries: list[SourceEntry]) -> None:
        self._entries = {entry.id: entry for entry in entries}
        if len(self._entries) != len(entries):
            raise ValueError("source IDs must be unique")

    @classmethod
    def from_yaml(cls, path: Path) -> SourceRegistry:
        """Load a source registry from a UTF-8 YAML file."""
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("sources"), list):
            raise ValueError("sources configuration must contain a sources list")
        return cls([SourceEntry.from_dict(item) for item in value["sources"]])

    def get(self, source_id: str) -> SourceEntry:
        """Return a named source or raise a useful error."""
        try:
            return self._entries[source_id]
        except KeyError as error:
            raise KeyError(f"unknown source ID: {source_id}") from error

    def iter_sources(
        self, source_ids: set[str] | None = None, statuses: set[str] | None = None
    ) -> Iterator[SourceEntry]:
        """Yield sources in configuration order after optional filters."""
        for source in self._entries.values():
            if source_ids is not None and source.id not in source_ids:
                continue
            if statuses is not None and source.status not in statuses:
                continue
            yield source

    def audit(self, allowed_licences: set[str]) -> list[dict[str, Any]]:
        """Return source decisions without downloading content."""
        decisions: list[dict[str, Any]] = []
        for source in self._entries.values():
            reasons: list[str] = []
            if source.dataset_level_licence not in allowed_licences:
                reasons.append("dataset_licence_not_allowlisted")
            if not source.per_record_licence:
                reasons.append("missing_per_record_licence")
            if not source.provenance_available:
                reasons.append("missing_provenance")
            if source.status not in {"approved", "approved_smoke"}:
                reasons.append(f"status_{source.status}")
            decisions.append(
                {
                    "source_id": source.id,
                    "official_url": source.official_url,
                    "version": source.version,
                    "licence": source.dataset_level_licence,
                    "status": "approved" if not reasons else "not_approved",
                    "reasons": reasons,
                }
            )
        return decisions
