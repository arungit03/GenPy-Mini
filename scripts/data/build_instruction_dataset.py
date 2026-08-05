"""Build the Exercism instruction dataset through the deterministic pairing adapter.

This is separate from the pretraining DatasetPipeline: Exercism's source status is
"approved_instruction_only", which is intentionally outside phase2.yaml's
ingestion.approved_statuses, so scripts/data/build_dataset.py can never select it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.data.exercism_adapter import (  # noqa: E402
    ExercismRejection,
    build_report,
    exact_deduplicate,
    iter_exercise_pairs,
    near_deduplicate,
)
from genpy.data.schemas import InstructionRecord  # noqa: E402
from genpy.data.sharding import write_shards  # noqa: E402
from genpy.data.source_registry import SourceRegistry  # noqa: E402
from genpy.data.splitting import split_instruction_records  # noqa: E402


def _leakage_check(records: list[InstructionRecord]) -> dict[str, Any]:
    hash_to_split: dict[str, str] = {}
    group_to_split: dict[str, str] = {}
    exact_cross_split = 0
    group_violations = 0
    for record in records:
        previous_split = hash_to_split.get(record.content_sha256)
        if previous_split is not None and previous_split != record.split:
            exact_cross_split += 1
        hash_to_split[record.content_sha256] = record.split
        previous_group_split = group_to_split.get(record.problem_family)
        if previous_group_split is not None and previous_group_split != record.split:
            group_violations += 1
        group_to_split[record.problem_family] = record.split
    return {
        "exact_cross_split_duplicates": exact_cross_split,
        "group_split_violations": group_violations,
        "passed": exact_cross_split == 0 and group_violations == 0,
    }


def main() -> int:
    """Build, split, shard, and report the Exercism instruction dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/phase2.yaml")
    parser.add_argument("--source", default="exercism_python")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sources_path = ROOT / str(config["sources_config"])
    registry = SourceRegistry.from_yaml(sources_path)
    source = registry.get(args.source)

    raw_root = ROOT / str(config["paths"]["raw"])
    normalization = config["normalization"]
    quality_config = config["quality"]
    dedup_config = config["deduplication"]
    ratios = {name: float(value) for name, value in config["splits"]["instruction"].items()}
    seed = int(config["seed"])
    sharding = config["sharding"]

    records: list[InstructionRecord] = []
    rejections: list[ExercismRejection] = []
    for item in iter_exercise_pairs(source, raw_root, normalization, quality_config):
        if isinstance(item, ExercismRejection):
            rejections.append(item)
        else:
            records.append(item)

    records, exact_duplicates = exact_deduplicate(records)
    records, near_duplicates, near_clusters = near_deduplicate(
        records,
        threshold=float(dedup_config["near_similarity_threshold"]),
        shingle_size=int(dedup_config["shingle_size"]),
    )
    records = list(split_instruction_records(records, ratios, seed))
    leakage = _leakage_check(records)

    cleaned_dir = ROOT / str(config["paths"]["cleaned"]) / "instruction"
    splits_dir = ROOT / str(config["paths"]["splits"]) / "instruction"
    write_shards(
        (record.to_dict() for record in records),
        cleaned_dir,
        maximum_uncompressed_bytes=int(sharding["maximum_uncompressed_bytes"]),
        compression_level=int(sharding["compression_level"]),
        config_hash="exercism-instruction-adapter-v1",
    )
    split_counts: Counter[str] = Counter()
    for split_name in ("train", "validation", "test"):

        def selected(name: str = split_name) -> Any:
            for record in records:
                if record.split == name:
                    split_counts[name] += 1
                    yield record.to_dict()

        write_shards(
            selected(),
            splits_dir / split_name,
            maximum_uncompressed_bytes=int(sharding["maximum_uncompressed_bytes"]),
            compression_level=int(sharding["compression_level"]),
            config_hash="exercism-instruction-adapter-v1",
        )

    report = build_report(
        records, rejections, exact_duplicates, near_duplicates, near_clusters,
        leakage["passed"], split_counts,
    )
    report["leakage"] = leakage
    report["source_revision"] = source.version
    report["source_archive_sha256"] = source.checksum_sha256

    reports_dir = ROOT / str(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "instruction_dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
