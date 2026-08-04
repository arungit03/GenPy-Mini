"""Train GenPy's own byte-level BPE from a Phase 2 corpus manifest."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from genpy.tokenizer.config import TokenizerConfig
from genpy.tokenizer.corpus import iter_manifest_training_text, prepare_corpus_manifest
from genpy.tokenizer.fingerprint import (
    atomic_write_json,
    canonical_sha256,
    sha256_file,
    write_checksum_file,
)
from genpy.tokenizer.validation import check_readiness


class TokenizerTrainingError(RuntimeError):
    """Raised when readiness or artifact safety prevents training."""


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _artifact_has_generated_files(path: Path) -> bool:
    return path.exists() and any(item.name != ".gitkeep" for item in path.iterdir())


def train_tokenizer(
    config: TokenizerConfig,
    *,
    mode: str,
    force: bool = False,
    maximum_bytes: int | None = None,
    maximum_records: int | None = None,
    source_ids: set[str] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Train, fingerprint, and atomically install one versioned tokenizer artifact."""
    if mode not in {"smoke", "production"}:
        raise TokenizerTrainingError("mode must be smoke or production")
    readiness = check_readiness(config)
    if mode == "production" and readiness.status != "READY_FOR_PRODUCTION_TOKENIZER":
        raise TokenizerTrainingError(f"production training blocked: {readiness.status}")
    if mode == "smoke" and readiness.status == "NOT_READY":
        raise TokenizerTrainingError("smoke training blocked: Phase 2 core readiness failed")
    if mode == "smoke" and config.tokenizer["status"] != "smoke":
        raise TokenizerTrainingError("smoke mode requires a smoke tokenizer config")
    if mode == "production" and int(config.tokenizer["vocab_size"]) != 16384:
        raise TokenizerTrainingError("production tokenizer must request 16384 tokens")

    summary = prepare_corpus_manifest(
        config,
        maximum_bytes=maximum_bytes,
        maximum_records=maximum_records,
        source_ids=source_ids,
    )
    if int(summary["selected_records"]) == 0:
        raise TokenizerTrainingError("tokenizer corpus manifest selected no records")
    artifact = output or config.artifact_path
    if _artifact_has_generated_files(artifact) and not force:
        raise FileExistsError(f"tokenizer artifact already exists: {artifact}")
    if force and config.tokenizer["status"] == "production":
        raise TokenizerTrainingError(
            "production artifacts cannot be overwritten under the same version"
        )

    building = artifact.with_name(artifact.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    started = time.perf_counter()
    model = models.BPE(dropout=None, unk_token=None)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=bool(config.tokenizer["add_prefix_space"]),
        use_regex=bool(config.tokenizer["use_regex"]),
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
        vocab_size=int(config.tokenizer["vocab_size"]),
        min_frequency=int(config.tokenizer["min_frequency"]),
        special_tokens=list(config.ordered_special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    manifest_path = config.resolve(str(config.corpus["manifest_path"]))
    tokenizer.train_from_iterator(
        iter_manifest_training_text(config, manifest_path), trainer=trainer
    )
    duration = time.perf_counter() - started
    actual_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    expected_vocab = int(config.tokenizer["expected_vocab_size"])
    if actual_vocab != expected_vocab:
        raise TokenizerTrainingError(
            f"actual vocabulary size {actual_vocab} differs from requested {expected_vocab}"
        )
    for name, special in config.special_tokens.items():
        if tokenizer.token_to_id(special.token) != special.id:
            raise TokenizerTrainingError(f"special token ID changed during training: {name}")

    tokenizer.save(str(building / "tokenizer.json"))
    saved_model = model.save(str(building))
    saved_names = {Path(path).name for path in saved_model}
    if saved_names != {"vocab.json", "merges.txt"}:
        raise TokenizerTrainingError("BPE model did not produce vocab.json and merges.txt")
    tokenizer_config = {
        "name": config.tokenizer["name"],
        "version": int(config.tokenizer["version"]),
        "algorithm": "byte_level_bpe",
        "vocab_size": actual_vocab,
        "add_prefix_space": bool(config.tokenizer["add_prefix_space"]),
        "use_regex": bool(config.tokenizer["use_regex"]),
        "unicode_normalization": "none",
        "lowercase": False,
        "dropout": 0.0,
        "context_length": int(config.tokenizer["context_length"]),
    }
    special_map = {
        name: {"token": special.token, "id": special.id}
        for name, special in config.special_tokens.items()
    }
    corpus_fingerprint = {
        "corpus_fingerprint": summary["corpus_fingerprint"],
        "manifest_sha256": summary["manifest_sha256"],
        "configuration_sha256": summary["configuration_sha256"],
        "selected_records": summary["selected_records"],
        "training_bytes": summary["training_bytes"],
    }
    atomic_write_json(building / "tokenizer_config.json", tokenizer_config)
    atomic_write_json(building / "special_tokens_map.json", special_map)
    atomic_write_json(building / "corpus_fingerprint.json", corpus_fingerprint)
    core_files = (
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "corpus_fingerprint.json",
    )
    core_checksums = {name: sha256_file(building / name) for name in core_files}
    fingerprint = canonical_sha256(
        {
            "core_checksums": core_checksums,
            "corpus_fingerprint": summary["corpus_fingerprint"],
            "configuration_sha256": config.config_hash,
            "special_tokens": special_map,
        }
    )
    phase2_report = json.loads(
        config.resolve(str(config.corpus["phase2_report"])).read_text(encoding="utf-8")
    )
    metadata: dict[str, Any] = {
        "tokenizer_name": config.tokenizer["name"],
        "version": int(config.tokenizer["version"]),
        "status": config.tokenizer["status"],
        "freeze_status": "not_frozen" if mode == "smoke" else "candidate_pending_validation",
        "algorithm": "byte_level_bpe",
        "requested_vocabulary_size": int(config.tokenizer["vocab_size"]),
        "actual_vocabulary_size": actual_vocab,
        "special_tokens": special_map,
        "minimum_frequency": int(config.tokenizer["min_frequency"]),
        "byte_level_settings": tokenizer_config,
        "training_seed": int(config.tokenizer["training_seed"]),
        "corpus_tier": "smoke" if mode == "smoke" else "production_candidate",
        "corpus_fingerprint": summary["corpus_fingerprint"],
        "tokenizer_fingerprint": fingerprint,
        "selected_records": summary["selected_records"],
        "training_bytes": summary["training_bytes"],
        "actual_mixture": summary["actual_mixture"],
        "source_composition": summary["source_composition"],
        "licence_composition": summary["licence_composition"],
        "phase2_dataset_version": phase2_report.get("corpus_version"),
        "phase2_pipeline_configuration_sha256": phase2_report.get("pipeline_configuration_sha256"),
        "configuration_sha256": config.config_hash,
        "python_version": platform.python_version(),
        "tokenizers_version": tokenizers.__version__,
        "operating_system": platform.system(),
        "creation_timestamp_utc": datetime.now(UTC).isoformat(),
        "training_duration_seconds": round(duration, 6),
        "thread_settings": {"workers": 1, "tokenizers_parallelism": False},
        "git_commit": _git_commit(config.project_root),
        "artifact_checksums": core_checksums,
        "known_limitations": [
            "Smoke corpus is one repository and has no instruction or held-out records."
            if mode == "smoke"
            else "Production acceptance is pending complete validation and evaluation.",
            "Cross-platform byte-identical reproducibility has not been claimed.",
            "Vocabulary security scanning cannot guarantee complete removal of sensitive data.",
        ],
    }
    atomic_write_json(building / "metadata.json", metadata)
    atomic_write_json(building / "evaluation.json", {"status": "not_evaluated"})
    all_files = (*core_files, "metadata.json", "evaluation.json")
    write_checksum_file(building, all_files)

    artifact.mkdir(parents=True, exist_ok=True)
    for source in building.iterdir():
        source.replace(artifact / source.name)
    building.rmdir()
    return metadata
