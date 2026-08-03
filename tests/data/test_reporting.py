"""Tests for acquisition report generation (JSON + Markdown)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from genpy.data.manifests import AcquisitionRecord, FileManifestRecord, build_manifest, write_manifest
from genpy.data.reporting import (
    RAW_DATA_DISCLAIMER,
    build_report,
    render_markdown_report,
    report_to_dict,
    write_json_report,
    write_markdown_report,
)
from genpy.data.schemas import AcquisitionSettings, DatasetSource, GovernanceReview, SourceLicense


def _acquired_source(source_id: str, *, spdx: str = "MIT", attribution_required: bool = False) -> DatasetSource:
    return DatasetSource(
        id=source_id,
        name=source_id,
        enabled=True,
        source_type="local_directory",
        location="/tmp/does-not-matter",
        description="Fixture source.",
        revision="v1",
        license=SourceLicense(
            declared_spdx=spdx,
            license_file="LICENSE",
            attribution_required=attribution_required,
            redistribution_allowed=True,
            commercial_use_allowed=True,
            modifications_allowed=True,
        ),
        governance=GovernanceReview(reviewed_by="Test", reviewed_on="2026-08-01", approval_status="approved"),
        acquisition=AcquisitionSettings(expected_sha256=None, maximum_download_bytes=1024, maximum_extracted_bytes=1024),
    )


def _write_fake_acquisition(
    tmp_path: Path,
    *,
    source_id: str,
    spdx: str = "MIT",
    attribution_required: bool = False,
    governance_override: bool = False,
    override_reason: str | None = None,
    corrupt_after_write: bool = False,
) -> None:
    sources_root = tmp_path / "raw" / "sources"
    manifests_dir = tmp_path / "manifests"

    destination = sources_root / source_id / "v1"
    destination.mkdir(parents=True)
    (destination / "file.py").write_text("content", encoding="utf-8")

    from genpy.data.checksums import sha256_file

    file_record = FileManifestRecord(
        relative_path="file.py",
        size_bytes=(destination / "file.py").stat().st_size,
        sha256=sha256_file(destination / "file.py"),
    )
    now = datetime.now(timezone.utc)
    record = AcquisitionRecord(
        started_at=now,
        completed_at=now,
        tool_version="0.1.0",
        governance_override=governance_override,
        override_reason=override_reason,
    )
    source = _acquired_source(source_id, spdx=spdx, attribution_required=attribution_required)
    manifest = build_manifest(source, "v1", [file_record], record)
    write_manifest(manifest, manifests_dir)

    if corrupt_after_write:
        (destination / "file.py").write_text("TAMPERED", encoding="utf-8")


def test_report_json_generation(tmp_path: Path) -> None:
    _write_fake_acquisition(tmp_path, source_id="alpha")
    report = build_report(tmp_path / "manifests", sources_root=tmp_path / "raw" / "sources")

    path = tmp_path / "reports" / "acquisition-report.json"
    write_json_report(report, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["sources"]["acquired"] == 1
    assert data["raw_data_disclaimer"] == RAW_DATA_DISCLAIMER


def test_report_markdown_generation(tmp_path: Path) -> None:
    _write_fake_acquisition(tmp_path, source_id="alpha")
    report = build_report(tmp_path / "manifests", sources_root=tmp_path / "raw" / "sources")

    path = tmp_path / "reports" / "acquisition-report.md"
    write_markdown_report(report, path)

    text = path.read_text(encoding="utf-8")
    assert "# GenPy-Mini Dataset Acquisition Report" in text
    assert RAW_DATA_DISCLAIMER in text
    assert "alpha" not in text  # nothing failed/overridden, so id shouldn't appear in those sections


def test_report_license_summary(tmp_path: Path) -> None:
    _write_fake_acquisition(tmp_path, source_id="alpha", spdx="MIT")
    _write_fake_acquisition(tmp_path, source_id="beta", spdx="MIT")
    _write_fake_acquisition(tmp_path, source_id="gamma", spdx="Apache-2.0")

    report = build_report(tmp_path / "manifests", sources_root=tmp_path / "raw" / "sources")
    assert report.license_summary == {"MIT": 2, "Apache-2.0": 1}


def test_report_attribution_required_sources(tmp_path: Path) -> None:
    _write_fake_acquisition(tmp_path, source_id="alpha", attribution_required=True)
    _write_fake_acquisition(tmp_path, source_id="beta", attribution_required=False)

    report = build_report(tmp_path / "manifests", sources_root=tmp_path / "raw" / "sources")
    assert report.attribution_required_sources == ("alpha",)


def test_report_override_summary(tmp_path: Path) -> None:
    _write_fake_acquisition(
        tmp_path, source_id="alpha", governance_override=True, override_reason="manually cleared"
    )
    report = build_report(tmp_path / "manifests", sources_root=tmp_path / "raw" / "sources")

    assert report.governance_overrides == 1
    assert report.override_details == ("alpha: manually cleared",)


def test_report_failed_acquisition_summary(tmp_path: Path) -> None:
    _write_fake_acquisition(tmp_path, source_id="alpha", corrupt_after_write=True)
    report = build_report(tmp_path / "manifests", sources_root=tmp_path / "raw" / "sources")

    assert report.failed_or_incomplete == ("alpha",)
    assert report.manifest_verification["alpha-v1"] is False


def test_report_fields_are_deterministic_across_runs(tmp_path: Path) -> None:
    _write_fake_acquisition(tmp_path, source_id="alpha")
    _write_fake_acquisition(tmp_path, source_id="beta")

    report_1 = build_report(tmp_path / "manifests", sources_root=tmp_path / "raw" / "sources")
    report_2 = build_report(tmp_path / "manifests", sources_root=tmp_path / "raw" / "sources")

    dict_1 = report_to_dict(report_1)
    dict_2 = report_to_dict(report_2)
    dict_1.pop("generated_at")
    dict_2.pop("generated_at")
    assert dict_1 == dict_2


def test_report_handles_empty_manifest_directory(tmp_path: Path) -> None:
    manifests_dir = tmp_path / "manifests"
    report = build_report(manifests_dir, sources_root=tmp_path / "raw" / "sources")

    assert report.acquired_sources == 0
    assert report.total_files == 0
    assert report.total_bytes == 0
    markdown = render_markdown_report(report)
    assert "_No acquired sources yet._" in markdown


def test_report_handles_corrupted_manifest_file(tmp_path: Path) -> None:
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True)
    (manifests_dir / "broken-v1.json").write_text("{not valid json", encoding="utf-8")

    report = build_report(manifests_dir, sources_root=tmp_path / "raw" / "sources")
    assert report.corrupted_manifests == ("broken-v1.json",)
    assert report.acquired_sources == 0
