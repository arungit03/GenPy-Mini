"""Exception hierarchy for Phase 2 dataset governance and acquisition.

All exceptions here derive from :class:`DatasetGovernanceError`. Messages
must be actionable and must never include credentials, tokens, or query
strings that could contain secrets -- see ``src/genpy/data/paths.py`` and
``src/genpy/data/acquisition.py`` for URL redaction helpers.
"""

from __future__ import annotations


class DatasetGovernanceError(Exception):
    """Base class for every Phase 2 dataset governance error."""


class SourceRegistryError(DatasetGovernanceError):
    """Raised when ``config/dataset_sources.yaml`` is missing, malformed, or invalid."""


class LicensePolicyError(DatasetGovernanceError):
    """Raised when ``config/license_policy.yaml`` is missing, malformed, or invalid."""


class AcquisitionError(DatasetGovernanceError):
    """Raised when acquiring a registered source fails."""


class ChecksumMismatchError(AcquisitionError):
    """Raised when a computed SHA-256 digest does not match the expected value."""


class StorageLimitError(AcquisitionError):
    """Raised when a download or extraction would exceed a configured byte limit."""


class UnsafePathError(AcquisitionError):
    """Raised when a path would escape an approved root or is otherwise unsafe."""


class UnsupportedSourceTypeError(AcquisitionError):
    """Raised when a ``source_type`` is not one of the supported source types."""
