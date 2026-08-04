"""Phase 2 readiness and tokenizer artifact validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from genpy.tokenizer.config import TokenizerConfig
from genpy.tokenizer.corpus import CorpusError, iter_family_items

ReadinessStatus = Literal[
    "READY_FOR_SMOKE_TOKENIZER",
    "READY_FOR_CANDIDATE_TOKENIZER",
    "READY_FOR_PRODUCTION_TOKENIZER",
    "NOT_READY",
]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One readiness condition."""

    name: str
    passed: bool
    detail: str
    production_required: bool = True


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    """Structured readiness status and evidence."""

    status: ReadinessStatus
    checks: tuple[CheckResult, ...]
    pretraining_records: int
    instruction_records: int
    available_serialized_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible readiness output."""
        return {
            "status": self.status,
            "checks": [asdict(check) for check in self.checks],
            "pretraining_records": self.pretraining_records,
            "instruction_records": self.instruction_records,
            "available_serialized_bytes": self.available_serialized_bytes,
        }


def _jsonl_loads(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return bool(lines) and all(isinstance(json.loads(line), dict) for line in lines)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def check_readiness(config: TokenizerConfig) -> ReadinessResult:
    """Validate Phase 2 evidence and classify tokenizer-training readiness."""
    checks: list[CheckResult] = []
    root = config.project_root
    source_manifest = root / "data/manifests/source_manifest.jsonl"
    licence_manifest = root / "data/manifests/licence_manifest.jsonl"
    checks.append(
        CheckResult(
            "source_manifest_loadable", _jsonl_loads(source_manifest), "safe source manifest"
        )
    )
    checks.append(
        CheckResult(
            "licence_manifest_loadable", _jsonl_loads(licence_manifest), "approved licence evidence"
        )
    )
    report_path = config.resolve(str(config.corpus["phase2_report"]))
    report: dict[str, Any] = {}
    if report_path.is_file():
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            report = loaded
    checks.append(CheckResult("phase2_report_loadable", bool(report), "Phase 2 machine report"))
    checks.append(
        CheckResult(
            "source_audit_passed",
            not report.get("failed_sources") and bool(report.get("source_distribution")),
            "selected sources completed without ingestion failures",
        )
    )
    leakage = report.get("leakage", {})
    checks.append(
        CheckResult(
            "no_cross_split_exact_duplicates",
            isinstance(leakage, dict)
            and leakage.get("passed") is True
            and leakage.get("exact_cross_split_duplicates") == 0,
            "Phase 2 leakage report",
        )
    )
    funnel = report.get("funnel", {})
    checks.append(
        CheckResult(
            "deduplication_complete",
            isinstance(funnel, dict)
            and "exact_deduplicated" in funnel
            and "near_deduplicated" in funnel,
            "exact and near-deduplication funnel stages present",
        )
    )
    checks.append(
        CheckResult(
            "quarantine_excluded",
            "quarantine" not in str(config.corpus["pretraining_train"]).lower()
            and "quarantine" not in str(config.corpus["instruction_train"]).lower(),
            "only split shard paths are configured",
        )
    )

    counts = {"pretraining": 0, "instruction": 0}
    available_bytes = 0
    try:
        for family in ("pretraining", "instruction"):
            for item in iter_family_items(config, family):
                counts[family] += 1
                available_bytes += item.serialized_bytes
        shard_check = counts["pretraining"] + counts["instruction"] > 0
    except (CorpusError, ValueError, OSError, json.JSONDecodeError):
        shard_check = False
    checks.append(
        CheckResult(
            "training_shards_valid",
            shard_check,
            "checksums, schemas, stable IDs, hashes, and train-only split verified",
        )
    )
    checks.append(
        CheckResult(
            "representative_instruction_data",
            counts["instruction"] > 0,
            f"instruction records available: {counts['instruction']}",
        )
    )
    validation_test_available = all(
        any((root / "data/splits" / family / split).glob("part-*.jsonl.zst"))
        for family in ("pretraining", "instruction")
        for split in ("validation", "test")
    )
    checks.append(
        CheckResult(
            "validation_and_test_available",
            validation_test_available,
            "held-out shards required for production evaluation",
        )
    )
    candidate_bytes = int(config.corpus["minimum_candidate_bytes"])
    production_bytes = int(config.corpus["minimum_production_bytes"])
    checks.append(
        CheckResult(
            "candidate_corpus_size",
            available_bytes >= candidate_bytes,
            f"{available_bytes} available bytes; {candidate_bytes} required",
        )
    )
    checks.append(
        CheckResult(
            "production_corpus_size",
            available_bytes >= production_bytes,
            f"{available_bytes} available bytes; {production_bytes} required",
        )
    )
    core_names = {
        "source_manifest_loadable",
        "licence_manifest_loadable",
        "phase2_report_loadable",
        "source_audit_passed",
        "no_cross_split_exact_duplicates",
        "deduplication_complete",
        "quarantine_excluded",
        "training_shards_valid",
    }
    core_passed = all(check.passed for check in checks if check.name in core_names)
    if not core_passed or counts["pretraining"] + counts["instruction"] < int(
        config.corpus["minimum_smoke_records"]
    ):
        status: ReadinessStatus = "NOT_READY"
    elif (
        available_bytes >= production_bytes
        and counts["pretraining"] > 0
        and counts["instruction"] > 0
        and (validation_test_available or not bool(config.corpus["require_validation_and_test"]))
    ):
        status = "READY_FOR_PRODUCTION_TOKENIZER"
    elif available_bytes >= candidate_bytes:
        status = "READY_FOR_CANDIDATE_TOKENIZER"
    else:
        status = "READY_FOR_SMOKE_TOKENIZER"
    return ReadinessResult(
        status, tuple(checks), counts["pretraining"], counts["instruction"], available_bytes
    )
