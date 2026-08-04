"""Phase 4 model, tokenizer, dataset, and packing readiness gate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from genpy.model.config import load_model_config
from genpy.model.parameter_count import count_parameters, validate_declared_tier
from genpy.tokenizer.tokenizer import GenPyTokenizer, TokenizerArtifactError
from genpy.training.packed_format import validate_packed_manifest
from genpy.training.packing import estimate_production_storage, load_packing_config

ReadinessStatus = Literal[
    "READY_FOR_SMOKE_MODEL", "READY_FOR_PACKING", "READY_FOR_PHASE5", "NOT_READY"
]


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    """One auditable Phase 4 readiness condition."""

    name: str
    passed: bool
    detail: str
    production_required: bool = True


@dataclass(frozen=True, slots=True)
class ModelReadiness:
    """Structured readiness result and honest current tier."""

    status: ReadinessStatus
    checks: tuple[ReadinessCheck, ...]
    model_name: str
    exact_parameters: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible readiness evidence."""
        return {
            "status": self.status,
            "model_name": self.model_name,
            "exact_parameters": self.exact_parameters,
            "checks": [asdict(check) for check in self.checks],
        }


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def check_model_readiness(config_path: Path, project_root: Path | None = None) -> ModelReadiness:
    """Classify smoke, packing, and Phase 5 readiness without weakening production gates."""
    config = load_model_config(config_path, project_root)
    root = config.project_root
    audit = count_parameters(config)
    validate_declared_tier(config, audit)
    checks: list[ReadinessCheck] = [
        ReadinessCheck("model_configuration_valid", True, config.config_hash),
        ReadinessCheck(
            "context_length_valid",
            config.context_length == (64 if config.is_smoke else 1024),
            str(config.context_length),
        ),
    ]
    dataset_report = _json_object(root / "data/reports/dataset_report.json")
    leakage = dataset_report.get("leakage", {})
    funnel = dataset_report.get("funnel", {})
    checks.extend(
        [
            ReadinessCheck(
                "phase2_status_known",
                bool(dataset_report.get("corpus_version")),
                str(dataset_report.get("corpus_version", "missing")),
            ),
            ReadinessCheck(
                "phase2_leakage_passed",
                isinstance(leakage, dict) and leakage.get("passed") is True,
                "deterministic split leakage evidence",
            ),
            ReadinessCheck(
                "phase2_deduplication_complete",
                isinstance(funnel, dict)
                and "exact_deduplicated" in funnel
                and "near_deduplicated" in funnel,
                "exact and near deduplication stages",
            ),
            ReadinessCheck(
                "no_quarantine_reference",
                "quarantine" not in str(config.tokenizer["artifact_path"]).lower(),
                "configured paths exclude quarantine",
            ),
            ReadinessCheck(
                "phase2_split_directories_exist",
                all(
                    (root / "data/splits" / family / split).is_dir()
                    for family in ("pretraining", "instruction")
                    for split in ("train", "validation", "test")
                ),
                "six isolated Phase 2 family/split directories",
            ),
        ]
    )
    tokenizer: GenPyTokenizer | None = None
    try:
        tokenizer = GenPyTokenizer.load(config.artifact_path)
    except (TokenizerArtifactError, OSError, ValueError):
        pass
    expected_fingerprint = str(config.tokenizer["fingerprint"])
    tokenizer_matches = (
        tokenizer is not None
        and tokenizer.fingerprint == expected_fingerprint
        and tokenizer.vocab_size == config.vocab_size
        and tokenizer.special_token_ids == config.tokenizer["special_token_ids"]
    )
    checks.extend(
        [
            ReadinessCheck(
                "tokenizer_artifact_valid", tokenizer is not None, str(config.artifact_path)
            ),
            ReadinessCheck(
                "tokenizer_contract_matches", tokenizer_matches, expected_fingerprint
            ),
            ReadinessCheck(
                "production_vocabulary_exact",
                config.is_smoke or (tokenizer_matches and config.vocab_size == 16384),
                str(config.vocab_size),
            ),
        ]
    )
    exact_counts = _json_object(root / "data/tokenizer/reports/exact_token_counts.json")
    counts_match = exact_counts.get("tokenizer_fingerprint") == expected_fingerprint
    checks.append(
        ReadinessCheck(
            "exact_token_counts_match", counts_match, "Phase 3 exact-count fingerprint"
        )
    )
    split_counts = exact_counts.get("splits", {})
    token_count = (
        sum(int(item.get("total_serialized_tokens", 0)) for item in split_counts.values())
        if isinstance(split_counts, dict)
        else 0
    )
    packing_path = root / "configs/data" / (
        "smoke_packing.yaml" if config.is_smoke else "packing.yaml"
    )
    try:
        packing_config = load_packing_config(packing_path, root)
        storage = estimate_production_storage(packing_config, token_count)
        storage_ok = bool(storage["within_configured_limit"]) and bool(
            storage["within_available_disk"]
        )
        storage_detail = str(storage["estimated_output_bytes"])
    except (OSError, ValueError):
        storage_ok = False
        storage_detail = "unavailable"
    checks.append(
        ReadinessCheck(
            "packing_storage_within_limits", storage_ok, storage_detail
        )
    )
    production_contracts = []
    for path in sorted((root / "configs/model").glob("genpy_*.yaml")):
        production_contracts.append(load_model_config(path, root).tokenizer["fingerprint"])
    contracts_match = len(set(production_contracts)) == 1 and production_contracts[0] != (
        "populated_after_training"
    )
    checks.append(
        ReadinessCheck(
            "production_model_fingerprints_locked",
            config.is_smoke or contracts_match,
            str(production_contracts[0] if production_contracts else "missing"),
        )
    )
    packed_ready = False
    if not config.is_smoke and tokenizer_matches:
        manifest = root / "data/packed/manifests/packing_manifest.json"
        if manifest.is_file():
            raw_manifest = _json_object(manifest)
            try:
                result = validate_packed_manifest(
                    manifest,
                    expected_fingerprint,
                    str(raw_manifest.get("packing_configuration_hash", "")),
                )
                packed_ready = result["passed"] and len(result["groups"]) == 6
            except (OSError, ValueError):
                packed_ready = False
    checks.append(
        ReadinessCheck(
            "all_production_splits_packed", packed_ready, "six isolated family/split groups"
        )
    )
    smoke_artifact_valid = False
    try:
        smoke = GenPyTokenizer.load(root / "artifacts/tokenizer/smoke")
        smoke_artifact_valid = smoke.vocab_size == 1024
    except (TokenizerArtifactError, OSError, ValueError):
        pass
    if config.is_smoke and tokenizer_matches:
        status: ReadinessStatus = "READY_FOR_SMOKE_MODEL"
    elif not config.is_smoke and tokenizer_matches and counts_match and contracts_match:
        status = "READY_FOR_PHASE5" if packed_ready else "READY_FOR_PACKING"
    elif smoke_artifact_valid:
        status = "READY_FOR_SMOKE_MODEL"
    else:
        status = "NOT_READY"
    return ModelReadiness(status, tuple(checks), config.name, audit.total_parameters)
