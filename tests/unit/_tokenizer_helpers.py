"""Original local-only Phase 3 fixture builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from genpy.data.sharding import write_shards
from tests.unit._helpers import make_record

SPECIAL = {
    "pad": {"token": "<|pad|>", "id": 0},
    "bos": {"token": "<|bos|>", "id": 1},
    "eos": {"token": "<|eos|>", "id": 2},
    "user": {"token": "<|user|>", "id": 3},
    "assistant": {"token": "<|assistant|>", "id": 4},
    "code": {"token": "<|code|>", "id": 5},
    "end": {"token": "<|end|>", "id": 6},
}


def fixture_code(marker: str = "base") -> str:
    """Return varied Python with enough pairs for a 500-token fixture vocabulary."""
    blocks = [
        f"def calculate_{marker}_{index}(value: int) -> int:\n"
        f"    \"\"\"Return a deterministic fixture result number {index}.\"\"\"\n"
        f"    adjusted_value = value + {index}\n"
        "    if adjusted_value % 2 == 0:\n"
        "        return adjusted_value * adjusted_value\n"
        "    return adjusted_value - 1\n"
        for index in range(80)
    ]
    return "\n".join(blocks)


def write_fixture_workspace(
    root: Path,
    *,
    marker: str = "base",
    contaminated_split: str | None = None,
    unsafe_quality: bool = False,
    vocab_size: int = 500,
) -> Path:
    """Create safe local Phase 2 evidence and a smoke-tokenizer config."""
    train_dir = root / "data/splits/pretraining/train"
    for split in ("validation", "test"):
        (root / "data/splits/pretraining" / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        (root / "data/splits/instruction" / split).mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(3):
        record = make_record(
            fixture_code(f"{marker}_{index}"),
            identity=f"fixture_{index}.py",
            group=f"fixture/project-{index}",
        )
        if contaminated_split:
            record.split = contaminated_split  # type: ignore[assignment]
        if unsafe_quality and index == 0:
            record.quality.secret_scan_passed = False
        records.append(record.to_dict())
    write_shards(
        records,
        train_dir,
        maximum_uncompressed_bytes=10_000_000,
        compression_level=1,
        config_hash="fixture-phase2-config",
    )
    manifests = root / "data/manifests"
    manifests.mkdir(parents=True)
    (manifests / "source_manifest.jsonl").write_text(
        json.dumps({"source_id": "test-fixture", "status": "approved_smoke"}) + "\n",
        encoding="utf-8",
    )
    (manifests / "licence_manifest.jsonl").write_text(
        json.dumps({"source_id": "test-fixture", "licence_spdx": "MIT"}) + "\n",
        encoding="utf-8",
    )
    reports = root / "data/reports"
    reports.mkdir(parents=True)
    report = {
        "corpus_version": "fixture-phase2-v1",
        "pipeline_configuration_sha256": "fixture-phase2-config",
        "failed_sources": [],
        "source_distribution": {"test-fixture": 3},
        "funnel": {"exact_deduplicated": 3, "near_deduplicated": 3},
        "leakage": {"passed": True, "exact_cross_split_duplicates": 0},
    }
    (reports / "dataset_report.json").write_text(json.dumps(report), encoding="utf-8")
    config: dict[str, Any] = {
        "schema_version": 1,
        "tokenizer": {
            "name": "genpy-fixture-smoke",
            "version": 1,
            "status": "smoke",
            "algorithm": "byte_level_bpe",
            "vocab_size": vocab_size,
            "expected_vocab_size": vocab_size,
            "min_frequency": 2,
            "add_prefix_space": False,
            "use_regex": True,
            "unicode_normalization": "none",
            "lowercase": False,
            "dropout": 0.0,
            "training_seed": 42,
            "include_special_tokens_in_vocab_size": True,
            "context_length": 1024,
            "artifact_path": "artifacts/tokenizer/fixture-smoke",
        },
        "special_tokens": SPECIAL,
        "corpus": {
            "phase2_config": "configs/data/phase2.yaml",
            "phase2_report": "data/reports/dataset_report.json",
            "pretraining_train": "data/splits/pretraining/train",
            "instruction_train": "data/splits/instruction/train",
            "manifest_path": "data/tokenizer/manifests/fixture.jsonl",
            "summary_path": "data/tokenizer/manifests/fixture_summary.json",
            "maximum_bytes": 2_000_000,
            "minimum_candidate_bytes": 100_000_000,
            "minimum_production_bytes": 500_000_000,
            "minimum_smoke_records": 1,
            "require_validation_and_test": False,
            "mixture": {"pretraining_weight": 0.85, "instruction_weight": 0.15},
        },
        "readiness": {"required_status": "READY_FOR_SMOKE_TOKENIZER"},
        "runtime": {"cpu_only": True, "workers": 1, "parallelism": False},
    }
    config_path = root / "configs/tokenizer/fixture.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path
