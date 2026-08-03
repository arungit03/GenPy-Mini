from __future__ import annotations

from pathlib import Path

import yaml

from genpy.data.pipeline import DatasetPipeline


def _write_configuration(tmp_path: Path) -> Path:
    source_fixture = Path("tests/fixtures/data/source").resolve()
    sources = {
        "schema_version": 1,
        "sources": [
            {
                "id": "local_fixture",
                "name": "Original local test fixture",
                "official_url": "https://example.invalid/genpy-fixture",
                "dataset_card_url": None,
                "version": "fixture-v1",
                "access_method": "local_directory",
                "local_path": str(source_fixture),
                "expected_download_size": "zero",
                "expected_extracted_size": "small",
                "expected_usable_records": "three",
                "estimated_disk_requirement": "less than 1 MB",
                "streaming_supported": True,
                "languages": ["Python"],
                "dataset_level_licence": "MIT",
                "per_record_licence": True,
                "provenance_available": True,
                "opt_out_supported": False,
                "attribution_required": True,
                "status": "approved_smoke",
                "review_notes": "Test-only fixture; never production data.",
                "repository": "genpy/test-fixture",
                "include_globs": ["*.py"],
            }
        ],
    }
    licences = {
        "allowlist": ["MIT"],
        "review_required": ["unknown"],
        "deny_families": ["GPL", "AGPL", "LGPL", "MPL"],
        "require_provenance": True,
        "require_repository_or_record_licence": True,
        "reject_dataset_file_conflicts": True,
    }
    config = {
        "pipeline_version": "test",
        "seed": 1337,
        "cpu_only": True,
        "sources_config": str(tmp_path / "sources.yaml"),
        "licences_config": str(tmp_path / "licenses.yaml"),
        "paths": {
            "raw": str(tmp_path / "raw"),
            "cleaned": str(tmp_path / "cleaned"),
            "splits": str(tmp_path / "splits"),
            "manifests": str(tmp_path / "manifests"),
            "quarantine": str(tmp_path / "quarantine"),
            "reports": str(tmp_path / "reports"),
            "work": str(tmp_path / "work"),
        },
        "ingestion": {
            "approved_statuses": ["approved_smoke"],
            "source_ids": ["local_fixture"],
            "resume": True,
            "global_record_limit": 20,
            "source_record_limit": 20,
            "maximum_download_bytes": 1000000,
            "minimum_free_disk_bytes": 1,
            "workers": 1,
        },
        "normalization": {"minimum_bytes": 10, "maximum_bytes": 100000, "final_newline": True},
        "path_filters": {"excluded_parts": ["__pycache__"], "include_tests": True},
        "quality": {
            "minimum_score": 0.4,
            "maximum_line_length": 300,
            "maximum_long_line_ratio": 0.2,
            "maximum_repeated_line_ratio": 0.5,
            "minimum_lexical_tokens": 4,
            "generated_markers": ["generated file", "do not edit"],
        },
        "deduplication": {
            "exact": True,
            "near": True,
            "near_similarity_threshold": 0.85,
            "shingle_size": 5,
            "minhash_permutations": 64,
            "lsh_bands": 16,
        },
        "splits": {
            "pretraining": {"train": 0.98, "validation": 0.01, "test": 0.01},
            "instruction": {"train": 0.8, "validation": 0.1, "test": 0.1},
        },
        "sharding": {
            "format": "jsonl.zst",
            "maximum_uncompressed_bytes": 10000,
            "compression_level": 1,
        },
        "reporting": {
            "rough_subword_bytes_per_token": 3.5,
            "corpus_version": "fixture-smoke-v1",
        },
    }
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(sources), encoding="utf-8")
    (tmp_path / "licenses.yaml").write_text(yaml.safe_dump(licences), encoding="utf-8")
    config_path = tmp_path / "phase2.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_local_end_to_end_pipeline_and_resume(tmp_path: Path) -> None:
    config_path = _write_configuration(tmp_path)
    first = DatasetPipeline(config_path, project_root=tmp_path).build()
    assert first["failed_sources"] == []
    assert first["total_records"] >= 1
    assert first["exact_duplicate_count"] >= 1
    assert first["secret_rejections"] == 1
    assert first["pii_rejections"] == 1
    assert first["rejections"]["invalid_python"] == 1
    assert first["leakage"]["passed"] is True
    assert (tmp_path / "reports/dataset_report.md").is_file()
    assert list((tmp_path / "splits/pretraining").rglob("*.jsonl.zst"))

    second = DatasetPipeline(config_path, project_root=tmp_path).build()
    assert second["resumed_from_completed_run"] is True
    assert second["pipeline_configuration_sha256"] == first["pipeline_configuration_sha256"]
