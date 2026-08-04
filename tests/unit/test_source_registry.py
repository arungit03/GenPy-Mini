from __future__ import annotations

from pathlib import Path

import pytest

from genpy.data.source_registry import SourceEntry, SourceRegistry


def test_current_registry_keeps_only_verified_click_source_approved() -> None:
    registry = SourceRegistry.from_yaml(Path("configs/data/sources.yaml"))
    decisions = {item["source_id"]: item for item in registry.audit({"BSD-3-Clause"})}

    assert decisions["pallets_click_8_1_8"]["status"] == "approved"
    assert decisions["pallets_click_8_1_8"]["reasons"] == []
    assert decisions["bigcode_the_stack_v2_1_0"]["status"] == "not_approved"


def test_github_source_requires_pinned_revision_checksum_and_matching_url() -> None:
    source = SourceEntry(
        id="unverified",
        name="Unverified source",
        official_url="https://github.com/example/project",
        dataset_card_url=None,
        version="main",
        access_method="github_archive",
        archive_url="https://codeload.github.com/example/project/zip/different",
        checksum_sha256="unknown",
        expected_download_size="unknown",
        expected_extracted_size="unknown",
        expected_usable_records="unknown",
        estimated_disk_requirement="unknown",
        streaming_supported=True,
        languages=("Python",),
        dataset_level_licence="MIT",
        per_record_licence=True,
        provenance_available=True,
        opt_out_supported=False,
        attribution_required=True,
        status="approved",
        review_notes="Fixture",
        repository="example/project",
    )

    decision = SourceRegistry([source]).audit({"MIT"})[0]

    assert decision["status"] == "not_approved"
    assert "revision_not_immutable_commit" in decision["reasons"]
    assert "missing_or_invalid_archive_checksum" in decision["reasons"]
    assert "archive_url_revision_mismatch" in decision["reasons"]
    with pytest.raises(ValueError, match="immutable commit"):
        source.validate()
