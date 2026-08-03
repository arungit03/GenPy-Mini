"""Human-readable and machine-readable acquisition audit reports.

Reports are built from whatever manifests exist under a manifests
directory, optionally combined with a loaded registry and license policy
for the "not yet acquired" categories (registered/enabled/approved/
review-required/rejected). They never claim acquired data is clean,
secret-free, deduplicated, or training-ready -- see ``RAW_DATA_DISCLAIMER``.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genpy.data.exceptions import DatasetGovernanceError
from genpy.data.licenses import LicensePolicy
from genpy.data.manifests import (
    read_manifest,
    sanitize_revision_for_filesystem,
    verify_manifest_against_directory,
)
from genpy.data.schemas import DatasetSourceRegistry
from genpy.data.source_registry import evaluate_registry

RAW_DATA_DISCLAIMER = (
    "Acquired data remains raw and has not yet passed cleaning, secret scanning, "
    "deduplication, quality filtering or train/test leakage controls."
)


@dataclasses.dataclass(frozen=True, slots=True)
class ManifestSummary:
    """A per-manifest summary used to build an :class:`AcquisitionReport`."""

    source_id: str
    revision: str
    file_count: int
    total_bytes: int
    declared_spdx: str
    approval_status: str
    attribution_required: bool
    forced: bool
    governance_override: bool
    override_reason: str | None
    verified: bool
    verification_problems: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class AcquisitionReport:
    """An aggregated snapshot of dataset acquisition state."""

    generated_at: datetime
    registered_sources: int
    enabled_sources: int
    approved_sources: int
    review_required_sources: int
    rejected_sources: int
    acquired_sources: int
    skipped_sources: int
    failed_sources: int
    governance_overrides: int
    total_files: int
    total_bytes: int
    license_summary: dict[str, int]
    attribution_required_sources: tuple[str, ...]
    override_details: tuple[str, ...]
    failed_or_incomplete: tuple[str, ...]
    manifest_verification: dict[str, bool]
    missing_expected_manifests: tuple[str, ...]
    corrupted_manifests: tuple[str, ...]


def _summarize_manifest(data: dict[str, Any], *, id_fallback: str, sources_root: Path) -> ManifestSummary:
    source_id = str(data.get("source_id", id_fallback))
    revision = str(data.get("resolved_revision", ""))

    summary_block = data.get("summary")
    summary_block = summary_block if isinstance(summary_block, dict) else {}
    license_block = data.get("license")
    license_block = license_block if isinstance(license_block, dict) else {}
    acquisition_block = data.get("acquisition")
    acquisition_block = acquisition_block if isinstance(acquisition_block, dict) else {}

    destination = sources_root / source_id / sanitize_revision_for_filesystem(revision)
    if destination.is_dir():
        verified, problems = verify_manifest_against_directory(data, destination)
    else:
        verified, problems = False, (f"acquired directory not found: {destination}",)

    return ManifestSummary(
        source_id=source_id,
        revision=revision,
        file_count=int(summary_block.get("file_count", 0)),
        total_bytes=int(summary_block.get("total_bytes", 0)),
        declared_spdx=str(license_block.get("declared_spdx", "UNKNOWN")),
        approval_status=str(license_block.get("approval_status", "unknown")),
        attribution_required=bool(license_block.get("attribution_required", False)),
        forced=bool(acquisition_block.get("forced", False)),
        governance_override=bool(acquisition_block.get("governance_override", False)),
        override_reason=acquisition_block.get("override_reason"),
        verified=verified,
        verification_problems=problems,
    )


def build_report(
    manifests_dir: Path,
    *,
    sources_root: Path | None = None,
    registry: DatasetSourceRegistry | None = None,
    policy: LicensePolicy | None = None,
) -> AcquisitionReport:
    """Build an :class:`AcquisitionReport` from manifests on disk.

    ``sources_root`` defaults to ``<manifests_dir>/../raw/sources``, matching
    the fixed Phase 2 repository layout. When ``registry`` and ``policy`` are
    both provided, the "not yet acquired" categories (registered, enabled,
    approved, review-required, rejected, missing-expected-manifests) are
    also populated; otherwise they are reported as zero / empty, since that
    information cannot be recovered from manifests alone.
    """
    resolved_sources_root = sources_root or (manifests_dir.parent / "raw" / "sources")

    manifest_files = sorted(manifests_dir.glob("*.json")) if manifests_dir.is_dir() else []
    summaries: list[ManifestSummary] = []
    corrupted: list[str] = []

    for manifest_file in manifest_files:
        try:
            data = read_manifest(manifest_file)
        except DatasetGovernanceError:
            corrupted.append(manifest_file.name)
            continue
        summaries.append(
            _summarize_manifest(data, id_fallback=manifest_file.stem, sources_root=resolved_sources_root)
        )

    total_files = sum(item.file_count for item in summaries)
    total_bytes = sum(item.total_bytes for item in summaries)

    license_summary: dict[str, int] = {}
    for item in summaries:
        license_summary[item.declared_spdx] = license_summary.get(item.declared_spdx, 0) + 1

    attribution_required = tuple(sorted(item.source_id for item in summaries if item.attribution_required))
    override_details = tuple(
        sorted(
            f"{item.source_id}: {item.override_reason}"
            for item in summaries
            if item.governance_override and item.override_reason
        )
    )
    failed_or_incomplete = tuple(sorted(item.source_id for item in summaries if not item.verified))
    manifest_verification = {f"{item.source_id}-{item.revision}": item.verified for item in summaries}
    acquired_ids = {item.source_id for item in summaries}

    registered_sources = enabled_sources = 0
    approved_sources = review_required_sources = rejected_sources = 0
    missing_expected_manifests: tuple[str, ...] = ()

    if registry is not None and policy is not None:
        evaluations = evaluate_registry(registry, policy)
        registered_sources = len(registry.sources)
        enabled_sources = sum(1 for source in registry.sources if source.enabled)
        approved_sources = sum(1 for e in evaluations if e.effective_status == "approved")
        review_required_sources = sum(1 for e in evaluations if e.effective_status == "review_required")
        rejected_sources = sum(1 for e in evaluations if e.effective_status == "rejected")
        missing_expected_manifests = tuple(
            sorted(
                e.source.id
                for e in evaluations
                if e.effective_status == "approved" and e.source.id not in acquired_ids
            )
        )

    return AcquisitionReport(
        generated_at=datetime.now(timezone.utc),
        registered_sources=registered_sources,
        enabled_sources=enabled_sources,
        approved_sources=approved_sources,
        review_required_sources=review_required_sources,
        rejected_sources=rejected_sources,
        acquired_sources=len(acquired_ids),
        skipped_sources=0,
        failed_sources=len(failed_or_incomplete),
        governance_overrides=len(override_details),
        total_files=total_files,
        total_bytes=total_bytes,
        license_summary=license_summary,
        attribution_required_sources=attribution_required,
        override_details=override_details,
        failed_or_incomplete=failed_or_incomplete,
        manifest_verification=manifest_verification,
        missing_expected_manifests=missing_expected_manifests,
        corrupted_manifests=tuple(sorted(corrupted)),
    )


def report_to_dict(report: AcquisitionReport) -> dict[str, Any]:
    """Render a report into its exact JSON-serializable shape."""
    return {
        "generated_at": report.generated_at.isoformat().replace("+00:00", "Z"),
        "raw_data_disclaimer": RAW_DATA_DISCLAIMER,
        "sources": {
            "registered": report.registered_sources,
            "enabled": report.enabled_sources,
            "approved": report.approved_sources,
            "review_required": report.review_required_sources,
            "rejected": report.rejected_sources,
            "acquired": report.acquired_sources,
            "skipped_this_run": report.skipped_sources,
            "failed_or_incomplete": report.failed_sources,
        },
        "totals": {
            "files": report.total_files,
            "bytes": report.total_bytes,
        },
        "license_summary": dict(sorted(report.license_summary.items())),
        "attribution_required_sources": list(report.attribution_required_sources),
        "governance_overrides": {
            "count": report.governance_overrides,
            "details": list(report.override_details),
        },
        "failed_or_incomplete_sources": list(report.failed_or_incomplete),
        "manifest_verification": dict(sorted(report.manifest_verification.items())),
        "missing_expected_manifests": list(report.missing_expected_manifests),
        "corrupted_manifests": list(report.corrupted_manifests),
    }


def write_json_report(report: AcquisitionReport, path: Path) -> None:
    """Write ``report`` as UTF-8 JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report_to_dict(report), indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def render_markdown_report(report: AcquisitionReport) -> str:
    """Render ``report`` as a human-readable Markdown document."""
    generated = report.generated_at.isoformat().replace("+00:00", "Z")
    lines: list[str] = [
        "# GenPy-Mini Dataset Acquisition Report",
        "",
        f"Generated: {generated}",
        "",
        f"> {RAW_DATA_DISCLAIMER}",
        "",
        "Skip/failure counts reflect what is inferable from manifests on disk: a "
        "manifest that fails verification against its acquired directory counts as "
        "failed/incomplete here. True mid-run skips are only visible in that run's "
        "own console output.",
        "",
        "## Source counts",
        "",
        "| Status | Count |",
        "| --- | --- |",
        f"| Registered | {report.registered_sources} |",
        f"| Enabled | {report.enabled_sources} |",
        f"| Approved | {report.approved_sources} |",
        f"| Review required | {report.review_required_sources} |",
        f"| Rejected | {report.rejected_sources} |",
        f"| Acquired (manifest present) | {report.acquired_sources} |",
        f"| Failed or incomplete | {report.failed_sources} |",
        "",
        "## Raw storage",
        "",
        f"- Total files: {report.total_files}",
        f"- Total bytes: {report.total_bytes} ({report.total_bytes / (1024**2):.1f} MiB)",
        "",
        "## License summary (declared SPDX, acquired sources)",
        "",
    ]

    if report.license_summary:
        lines += ["| SPDX | Acquired sources |", "| --- | --- |"]
        lines += [f"| {spdx} | {count} |" for spdx, count in sorted(report.license_summary.items())]
    else:
        lines.append("_No acquired sources yet._")

    lines += ["", "## Attribution required", ""]
    lines += (
        [f"- {source_id}" for source_id in report.attribution_required_sources]
        if report.attribution_required_sources
        else ["_None._"]
    )

    lines += ["", "## Governance overrides used", ""]
    lines += [f"- {detail}" for detail in report.override_details] if report.override_details else ["_None._"]

    lines += ["", "## Failed or incomplete acquisitions", ""]
    lines += (
        [f"- {source_id}" for source_id in report.failed_or_incomplete]
        if report.failed_or_incomplete
        else ["_None._"]
    )

    lines += ["", "## Missing expected manifests (approved but not yet acquired)", ""]
    lines += (
        [f"- {source_id}" for source_id in report.missing_expected_manifests]
        if report.missing_expected_manifests
        else ["_None._"]
    )

    if report.corrupted_manifests:
        lines += ["", "## Corrupted manifest files", ""]
        lines += [f"- {name}" for name in report.corrupted_manifests]

    lines.append("")
    return "\n".join(lines)


def write_markdown_report(report: AcquisitionReport, path: Path) -> None:
    """Write ``report`` as Markdown to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(report), encoding="utf-8")
