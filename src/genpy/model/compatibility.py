"""Safe project-checkpoint compatibility and checksum enforcement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from genpy.model.config import ModelConfig
from genpy.model.parameter_count import count_parameters
from genpy.model.transformer import GenPyForCausalLM
from genpy.tokenizer.fingerprint import atomic_write_json, sha256_file

CHECKPOINT_FIELDS = {
    "format_version", "model_name", "architecture", "model_configuration_hash",
    "parameter_count", "tokenizer_name", "tokenizer_version", "tokenizer_fingerprint",
    "vocabulary_size", "special_token_ids", "context_length", "pytorch_version",
    "random_seed", "weight_dtype", "training_phase", "weights_sha256",
}


class CompatibilityError(RuntimeError):
    """Raised when model, tokenizer, or checkpoint metadata differ."""


def checkpoint_metadata_values(config: ModelConfig) -> dict[str, Any]:
    """Return compatibility values independent of a particular weights file."""
    return {
        "format_version": 1,
        "model_name": config.name,
        "architecture": config.architecture,
        "model_configuration_hash": config.config_hash,
        "parameter_count": count_parameters(config).total_parameters,
        "tokenizer_name": config.tokenizer["name"],
        "tokenizer_version": config.tokenizer["version"],
        "tokenizer_fingerprint": config.tokenizer["fingerprint"],
        "vocabulary_size": config.vocab_size,
        "special_token_ids": config.tokenizer["special_token_ids"],
        "context_length": config.context_length,
        "random_seed": config.seed,
    }


def checkpoint_metadata(config: ModelConfig, weights_path: Path) -> dict[str, Any]:
    """Build mandatory metadata for trusted project-created tensor state."""
    return {
        **checkpoint_metadata_values(config),
        "pytorch_version": torch.__version__,
        "weight_dtype": "float32",
        "training_phase": "phase4_smoke",
        "weights_sha256": sha256_file(weights_path),
    }


def save_project_state(model: GenPyForCausalLM, directory: Path) -> dict[str, Any]:
    """Save a trusted tensor-only state and atomic JSON metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "model_state.pt"
    temporary = directory / "model_state.pt.tmp"
    torch.save(model.state_dict(), temporary)
    temporary.replace(weights)
    metadata = checkpoint_metadata(model.config, weights)
    atomic_write_json(directory / "metadata.json", metadata)
    return metadata


def validate_checkpoint_metadata(config: ModelConfig, metadata: dict[str, Any]) -> None:
    """Reject incomplete or incompatible checkpoint metadata before state loading."""
    missing = CHECKPOINT_FIELDS - set(metadata)
    if missing:
        raise CompatibilityError(f"checkpoint metadata missing: {', '.join(sorted(missing))}")
    for key, value in checkpoint_metadata_values(config).items():
        if metadata.get(key) != value:
            raise CompatibilityError(f"checkpoint compatibility mismatch: {key}")


def load_project_state(model: GenPyForCausalLM, directory: Path) -> None:
    """Load only a checksum-verified state created by this project."""
    weights = directory / "model_state.pt"
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompatibilityError("checkpoint metadata cannot be loaded") from error
    if not isinstance(metadata, dict):
        raise CompatibilityError("checkpoint metadata must be an object")
    validate_checkpoint_metadata(model.config, metadata)
    if metadata["weights_sha256"] != sha256_file(weights):
        raise CompatibilityError("checkpoint weights checksum failed")
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise CompatibilityError("checkpoint state cannot be loaded") from error
    if model.token_embedding.weight is not model.lm_head.weight:
        raise CompatibilityError("weight tying did not survive state loading")
