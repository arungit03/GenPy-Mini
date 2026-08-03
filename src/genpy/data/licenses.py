"""License policy loading and evaluation.

This module answers exactly one question: given an SPDX identifier, what
does ``config/license_policy.yaml`` say about it (``allowed``,
``review_required``, or ``blocked``)? It does not, and cannot, tell you
whether a specific repository is actually safe to train on -- see the
mandatory disclaimer in the policy file and in
``docs/dataset-acquisition.md``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Final

import yaml

from genpy.data.exceptions import LicensePolicyError

VALID_LICENSE_STATUSES: Final[tuple[str, ...]] = ("allowed", "review_required", "blocked")

_REQUIRED_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "disclaimer",
    "default_status",
    "allowed",
    "review_required",
    "blocked",
)
_KNOWN_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(_REQUIRED_TOP_LEVEL_FIELDS)


@dataclasses.dataclass(frozen=True, slots=True)
class LicensePolicy:
    """A conservative, configurable SPDX allow/review/block policy.

    This is a starting point for human review, not a legal determination.
    Being in ``allowed`` means the license *family* is broadly compatible
    with training use in common cases -- it does not mean every repository
    carrying that SPDX identifier has been individually verified. See
    :meth:`evaluate` and the policy file's own ``disclaimer`` field.
    """

    schema_version: int
    disclaimer: str
    default_status: str
    allowed: frozenset[str]
    review_required: frozenset[str]
    blocked: frozenset[str]

    def __post_init__(self) -> None:
        if not self.disclaimer.strip():
            raise LicensePolicyError("license_policy.disclaimer must not be empty.")
        if self.default_status not in VALID_LICENSE_STATUSES:
            raise LicensePolicyError(
                f"license_policy.default_status must be one of {VALID_LICENSE_STATUSES}, "
                f"got {self.default_status!r}."
            )
        overlap = (self.allowed & self.review_required) | (self.allowed & self.blocked)
        overlap |= self.review_required & self.blocked
        if overlap:
            raise LicensePolicyError(
                f"license_policy: the following SPDX identifiers appear in more than one "
                f"status list: {sorted(overlap)}."
            )

    def evaluate(self, spdx_id: str) -> str:
        """Return ``allowed``, ``review_required``, or ``blocked`` for ``spdx_id``.

        An identifier absent from every list falls back to
        :attr:`default_status`. This function never treats mere public
        availability of code as a license grant -- see the module
        docstring.
        """
        normalized = spdx_id.strip()
        if normalized in self.blocked:
            return "blocked"
        if normalized in self.allowed:
            return "allowed"
        if normalized in self.review_required:
            return "review_required"
        return self.default_status


def load_license_policy(path: Path) -> LicensePolicy:
    """Load, validate, and return the license policy at ``path``.

    Raises:
        LicensePolicyError: if the file is missing, is not valid YAML, does
            not contain a mapping, is missing a required field, contains an
            unknown top-level field, or fails a consistency check.
    """
    if not path.exists():
        raise LicensePolicyError(f"License policy file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LicensePolicyError(f"Could not read license policy file {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise LicensePolicyError(f"License policy file {path} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise LicensePolicyError(f"License policy file {path} must contain a mapping at the top level.")

    missing = [field for field in _REQUIRED_TOP_LEVEL_FIELDS if field not in data]
    if missing:
        raise LicensePolicyError(
            f"License policy file {path} is missing required field(s): {', '.join(missing)}"
        )

    unknown = sorted(set(data) - _KNOWN_TOP_LEVEL_FIELDS)
    if unknown:
        raise LicensePolicyError(
            f"License policy file {path} has unknown top-level field(s): {', '.join(unknown)}"
        )

    try:
        return LicensePolicy(
            schema_version=int(data["schema_version"]),
            disclaimer=str(data["disclaimer"]),
            default_status=str(data["default_status"]),
            allowed=_as_spdx_set(data["allowed"], "allowed", path),
            review_required=_as_spdx_set(data["review_required"], "review_required", path),
            blocked=_as_spdx_set(data["blocked"], "blocked", path),
        )
    except LicensePolicyError:
        raise
    except (TypeError, ValueError) as exc:
        raise LicensePolicyError(f"License policy file {path} is malformed: {exc}") from exc


def _as_spdx_set(raw: Any, field: str, source: Path) -> frozenset[str]:
    if not isinstance(raw, list):
        raise LicensePolicyError(f"License policy file {source}: '{field}' must be a list.")
    values = [str(item).strip() for item in raw]
    if any(not value for value in values):
        raise LicensePolicyError(f"License policy file {source}: '{field}' must not contain blank entries.")
    return frozenset(values)
