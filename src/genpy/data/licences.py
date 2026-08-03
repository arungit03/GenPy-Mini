"""Conservative SPDX allowlist enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class LicencePolicy:
    """Project-specific licence decisions; this is not legal advice."""

    allowlist: frozenset[str]
    review_required: frozenset[str]
    deny_families: tuple[str, ...]
    require_provenance: bool
    require_repository_or_record_licence: bool
    reject_dataset_file_conflicts: bool

    @classmethod
    def from_yaml(cls, path: Path) -> LicencePolicy:
        """Load licence policy from YAML."""
        value: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("licence configuration must be an object")
        return cls(
            allowlist=frozenset(str(item) for item in value.get("allowlist", [])),
            review_required=frozenset(str(item) for item in value.get("review_required", [])),
            deny_families=tuple(str(item) for item in value.get("deny_families", [])),
            require_provenance=bool(value.get("require_provenance", True)),
            require_repository_or_record_licence=bool(
                value.get("require_repository_or_record_licence", True)
            ),
            reject_dataset_file_conflicts=bool(value.get("reject_dataset_file_conflicts", True)),
        )

    def decision(
        self,
        licence_spdx: str | None,
        *,
        provenance_available: bool,
        dataset_licence: str | None = None,
    ) -> tuple[bool, str | None]:
        """Return an allow decision and a machine-readable rejection reason."""
        if self.require_provenance and not provenance_available:
            return False, "missing_provenance"
        if not licence_spdx or licence_spdx in self.review_required:
            return False, "unknown_licence"
        upper = licence_spdx.upper()
        if any(family.upper() in upper for family in self.deny_families):
            return False, "unknown_licence"
        if licence_spdx not in self.allowlist:
            return False, "unknown_licence"
        if (
            self.reject_dataset_file_conflicts
            and dataset_licence
            and dataset_licence not in self.allowlist
            and dataset_licence not in {"mixed", "other"}
        ):
            return False, "unknown_licence"
        return True, None
