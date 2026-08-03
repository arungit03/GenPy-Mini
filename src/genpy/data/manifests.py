"""Provenance manifest construction, writing, and verification.

Every acquired source gets one deterministic JSON manifest in
``data/manifests/<source-id>-<revision>.json``. Manifests are the audit
trail for Phase 2: they record exactly what was acquired, from where
(credentials redacted), under what license determination, and with what
per-file hashes -- without ever claiming the acquired content is clean,
secret-free, or training-ready.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

from genpy.data.checksums import FileFingerprint, directory_digest, sha256_file
from genpy.data.exceptions import AcquisitionError, DatasetGovernanceError
from genpy.data.schemas import DatasetSource

MANIFEST_SCHEMA_VERSION: Final[int] = 1


def redact_url_credentials(url: str) -> str:
    """Strip userinfo, query string, and fragment from a URL for safe storage/logging.

    Query strings are removed entirely (not just parsed for credentials)
    because tokens are commonly smuggled as query parameters (e.g.
    ``?token=...``). Values that are not URLs (e.g. local filesystem paths)
    are returned unchanged.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable-location>"

    if not parts.scheme or not parts.netloc:
        return url

    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"

    redacted = parts._replace(netloc=netloc, query="", fragment="")
    return urlunsplit(redacted)


@dataclasses.dataclass(frozen=True, slots=True)
class FileManifestRecord:
    """One file's identity within a source's provenance manifest."""

    relative_path: str
    size_bytes: int
    sha256: str

    @staticmethod
    def from_fingerprint(fingerprint: FileFingerprint) -> FileManifestRecord:
        """Build a manifest record from a generic filesystem fingerprint."""
        return FileManifestRecord(
            relative_path=fingerprint.relative_path,
            size_bytes=fingerprint.size_bytes,
            sha256=fingerprint.sha256,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class AcquisitionRecord:
    """Metadata about the acquisition run that produced a manifest."""

    started_at: datetime
    completed_at: datetime
    tool_version: str
    forced: bool = False
    governance_override: bool = False
    override_reason: str | None = None

    def __post_init__(self) -> None:
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise AcquisitionError("AcquisitionRecord timestamps must be timezone-aware (UTC).")
        if self.governance_override and not (self.override_reason and self.override_reason.strip()):
            raise AcquisitionError(
                "AcquisitionRecord.override_reason is required when governance_override is true."
            )


@dataclasses.dataclass(frozen=True, slots=True)
class SourceManifest:
    """The complete provenance manifest for one acquired source."""

    schema_version: int
    source_id: str
    source_name: str
    source_type: str
    configured_location: str
    resolved_revision: str
    license_declared_spdx: str
    license_approval_status: str
    license_attribution_required: bool
    license_notes: str
    acquisition: AcquisitionRecord
    files: tuple[FileManifestRecord, ...]

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return sum(record.size_bytes for record in self.files)


def build_manifest(
    source: DatasetSource,
    resolved_revision: str,
    files: Sequence[FileManifestRecord],
    acquisition: AcquisitionRecord,
) -> SourceManifest:
    """Assemble a :class:`SourceManifest` with deterministically sorted files."""
    sorted_files = tuple(sorted(files, key=lambda record: record.relative_path))
    return SourceManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        source_id=source.id,
        source_name=source.name,
        source_type=source.source_type,
        configured_location=redact_url_credentials(source.location),
        resolved_revision=resolved_revision,
        license_declared_spdx=source.license.declared_spdx,
        license_approval_status=source.governance.approval_status,
        license_attribution_required=source.license.attribution_required,
        license_notes=source.license.notes,
        acquisition=acquisition,
        files=sorted_files,
    )


def _format_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def manifest_to_dict(manifest: SourceManifest) -> dict[str, Any]:
    """Render a manifest into the exact JSON-serializable shape written to disk."""
    fingerprints = tuple(
        FileFingerprint(relative_path=f.relative_path, size_bytes=f.size_bytes, sha256=f.sha256)
        for f in manifest.files
    )

    acquisition_dict: dict[str, Any] = {
        "started_at": _format_utc(manifest.acquisition.started_at),
        "completed_at": _format_utc(manifest.acquisition.completed_at),
        "tool_version": manifest.acquisition.tool_version,
        "forced": manifest.acquisition.forced,
        "governance_override": manifest.acquisition.governance_override,
    }
    if manifest.acquisition.override_reason:
        acquisition_dict["override_reason"] = manifest.acquisition.override_reason

    return {
        "schema_version": manifest.schema_version,
        "source_id": manifest.source_id,
        "source_name": manifest.source_name,
        "source_type": manifest.source_type,
        "configured_location": manifest.configured_location,
        "resolved_revision": manifest.resolved_revision,
        "license": {
            "declared_spdx": manifest.license_declared_spdx,
            "approval_status": manifest.license_approval_status,
            "attribution_required": manifest.license_attribution_required,
            "notes": manifest.license_notes,
        },
        "acquisition": acquisition_dict,
        "summary": {
            "file_count": manifest.file_count,
            "total_bytes": manifest.total_bytes,
        },
        "files": [
            {"relative_path": f.relative_path, "size_bytes": f.size_bytes, "sha256": f.sha256}
            for f in manifest.files
        ],
        "manifest_digest": directory_digest(fingerprints),
    }


def sanitize_revision_for_filesystem(revision: str) -> str:
    """Make a revision string safe to use as a path segment or filename component.

    Used consistently for both the manifest filename and the acquired
    source's destination directory name, so the two stay paired.
    """
    return revision.replace("/", "_").replace("\\", "_")


def manifest_path_for(manifests_dir: Path, source_id: str, revision: str) -> Path:
    """Return the deterministic manifest path for a source id and revision."""
    return manifests_dir / f"{source_id}-{sanitize_revision_for_filesystem(revision)}.json"


def write_manifest(manifest: SourceManifest, manifests_dir: Path) -> Path:
    """Write ``manifest`` as UTF-8 JSON and return the path written to.

    Writes to a temporary file in the same directory and renames it into
    place, so a reader never observes a partially written manifest.
    """
    manifests_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path_for(manifests_dir, manifest.source_id, manifest.resolved_revision)
    payload = manifest_to_dict(manifest)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    temp_path = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)
    return path


def read_manifest(path: Path) -> dict[str, Any]:
    """Read and parse a manifest JSON file into a plain dict."""
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetGovernanceError(f"Could not read manifest {path}: {exc}") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise DatasetGovernanceError(f"Manifest {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise DatasetGovernanceError(f"Manifest {path} must contain a JSON object at the top level.")

    return data


def verify_manifest_against_directory(
    manifest_data: dict[str, Any], source_root: Path
) -> tuple[bool, tuple[str, ...]]:
    """Verify a manifest's file records against the files actually on disk.

    Returns ``(ok, problems)``. ``ok`` is ``True`` only when every listed
    file exists with a matching size and SHA-256, and no unlisted files are
    present under ``source_root``.
    """
    problems: list[str] = []
    entries = manifest_data.get("files", [])
    if not isinstance(entries, list):
        return False, ("manifest 'files' field is not a list",)

    manifest_paths: set[str] = set()
    for entry in entries:
        relative_path = str(entry.get("relative_path", ""))
        manifest_paths.add(relative_path)
        file_path = source_root / relative_path

        if not file_path.is_file():
            problems.append(f"missing file: {relative_path}")
            continue

        expected_size = entry.get("size_bytes")
        actual_size = file_path.stat().st_size
        if actual_size != expected_size:
            problems.append(f"size mismatch for {relative_path}: expected {expected_size}, got {actual_size}")
            continue

        expected_sha256 = str(entry.get("sha256", ""))
        actual_sha256 = sha256_file(file_path)
        if actual_sha256 != expected_sha256:
            problems.append(f"sha256 mismatch for {relative_path}")

    actual_paths = {path.relative_to(source_root).as_posix() for path in source_root.rglob("*") if path.is_file()}
    extra = sorted(actual_paths - manifest_paths)
    if extra:
        problems.append(f"unexpected file(s) not recorded in manifest: {', '.join(extra)}")

    return (not problems, tuple(problems))
