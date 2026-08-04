"""Strict configuration loading for GenPy Transformer profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from genpy.tokenizer.config import SPECIAL_TOKEN_NAMES

MODEL_KEYS = {
    "name",
    "purpose",
    "architecture",
    "vocab_size",
    "context_length",
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dimension",
    "intermediate_size",
    "activation",
    "normalization",
    "norm_epsilon",
    "positional_encoding",
    "rope_base",
    "attention",
    "tie_word_embeddings",
    "attention_dropout",
    "residual_dropout",
    "embedding_dropout",
    "use_bias",
    "initializer_std",
    "gradient_checkpointing",
    "initialization",
    "parameter_count_note",
}
TOKENIZER_KEYS = {
    "name",
    "version",
    "type",
    "vocab_size",
    "trained",
    "artifact_path",
    "fingerprint",
    "special_tokens",
    "special_token_ids",
}
TRAINING_KEYS = {"seed", "phase"}


class ModelConfigError(ValueError):
    """Raised before allocation when a model contract is invalid."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Validated model, tokenizer, and initialization settings."""

    path: Path
    project_root: Path
    name: str
    architecture: str
    vocab_size: int
    context_length: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dimension: int
    intermediate_size: int
    norm_epsilon: float
    rope_base: float
    attention_dropout: float
    residual_dropout: float
    embedding_dropout: float
    use_bias: bool
    tie_word_embeddings: bool
    initializer_std: float
    gradient_checkpointing: bool
    seed: int
    tokenizer: dict[str, Any]
    config_hash: str
    is_smoke: bool

    @property
    def artifact_path(self) -> Path:
        """Resolve the tokenizer artifact from the repository root."""
        return self.project_root / str(self.tokenizer["artifact_path"])


def _project_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def _require_keys(value: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(value) - allowed
    missing = allowed - set(value)
    optional = {"purpose"} if section == "model" else set()
    missing -= optional
    if unknown:
        raise ModelConfigError(f"unknown {section} keys: {', '.join(sorted(unknown))}")
    if missing:
        raise ModelConfigError(f"missing {section} keys: {', '.join(sorted(missing))}")


def load_model_config(path: Path, project_root: Path | None = None) -> ModelConfig:
    """Load a complete model profile and validate all locked architecture decisions."""
    resolved = path.resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"model", "tokenizer", "training"}:
        raise ModelConfigError("model config requires model, tokenizer, and training mappings")
    model = raw["model"]
    tokenizer = raw["tokenizer"]
    training = raw["training"]
    if not all(isinstance(value, dict) for value in (model, tokenizer, training)):
        raise ModelConfigError("configuration sections must be mappings")
    _require_keys(model, MODEL_KEYS, "model")
    _require_keys(tokenizer, TOKENIZER_KEYS, "tokenizer")
    _require_keys(training, TRAINING_KEYS, "training")
    integers = (
        "vocab_size",
        "context_length",
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dimension",
        "intermediate_size",
    )
    if any(int(model[key]) <= 0 for key in integers):
        raise ModelConfigError("model dimensions must be positive")
    hidden = int(model["hidden_size"])
    heads = int(model["num_attention_heads"])
    kv_heads = int(model["num_key_value_heads"])
    if hidden % heads or int(model["head_dimension"]) != hidden // heads:
        raise ModelConfigError("hidden size, heads, and head dimension are inconsistent")
    if heads % kv_heads or kv_heads != heads:
        raise ModelConfigError("standard multi-head attention requires equal Q and KV heads")
    if int(model["head_dimension"]) % 2:
        raise ModelConfigError("RoPE requires an even head dimension")
    if model["architecture"] != "decoder_only_transformer":
        raise ModelConfigError("architecture must be decoder_only_transformer")
    required_choices = {
        "activation": "swiglu",
        "normalization": "rmsnorm",
        "positional_encoding": "rope",
        "attention": "causal_self_attention",
        "initialization": "random",
    }
    for key, expected in required_choices.items():
        if model[key] != expected:
            raise ModelConfigError(f"{key} must be {expected}")
    if model["use_bias"] is not False or model["tie_word_embeddings"] is not True:
        raise ModelConfigError("GenPy requires bias-free layers and tied embeddings")
    for key in ("attention_dropout", "residual_dropout", "embedding_dropout"):
        if not 0.0 <= float(model[key]) < 1.0:
            raise ModelConfigError(f"{key} must be in [0, 1)")
    is_smoke = str(model.get("purpose", "")) == "phase4_cpu_correctness_only"
    expected_vocab = int(tokenizer["vocab_size"]) if is_smoke else 16384
    context_valid = (
        1 <= int(model["context_length"]) <= 64
        if is_smoke
        else int(model["context_length"]) == 1024
    )
    if int(model["vocab_size"]) != expected_vocab or not context_valid:
        raise ModelConfigError("vocabulary or context length violates the model tier contract")
    if int(tokenizer["vocab_size"]) != int(model["vocab_size"]):
        raise ModelConfigError("model and tokenizer vocabulary sizes differ")
    expected_ids = dict(zip(SPECIAL_TOKEN_NAMES, range(7), strict=True))
    if tokenizer["special_token_ids"] != expected_ids:
        raise ModelConfigError("special-token IDs differ from the locked contract")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    root = (project_root or _project_root(resolved)).resolve()
    return ModelConfig(
        path=resolved,
        project_root=root,
        name=str(model["name"]),
        architecture=str(model["architecture"]),
        vocab_size=int(model["vocab_size"]),
        context_length=int(model["context_length"]),
        num_layers=int(model["num_layers"]),
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dimension=int(model["head_dimension"]),
        intermediate_size=int(model["intermediate_size"]),
        norm_epsilon=float(model["norm_epsilon"]),
        rope_base=float(model["rope_base"]),
        attention_dropout=float(model["attention_dropout"]),
        residual_dropout=float(model["residual_dropout"]),
        embedding_dropout=float(model["embedding_dropout"]),
        use_bias=bool(model["use_bias"]),
        tie_word_embeddings=bool(model["tie_word_embeddings"]),
        initializer_std=float(model["initializer_std"]),
        gradient_checkpointing=bool(model["gradient_checkpointing"]),
        seed=int(training["seed"]),
        tokenizer=dict(tokenizer),
        config_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        is_smoke=is_smoke,
    )
