"""Validated dataclass configuration for the Checkpoint 1 foundation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

PathLike = str | Path


def _mapping(raw: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"'{name}' must be a mapping")
    return raw


def _required(section: Mapping[str, Any], name: str, section_name: str) -> Any:
    if name not in section:
        raise ValueError(f"Missing required field '{section_name}.{name}'")
    return section[name]


def _check_keys(section: Mapping[str, Any], allowed: set[str], section_name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"Unknown field(s) in '{section_name}': {', '.join(unknown)}")


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"'{field}' must be a positive integer")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"'{field}' must be a non-negative integer")
    return value


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value < 1.0:
        raise ValueError(f"'{field}' must satisfy 0.0 <= value < 1.0")
    return float(value)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    max_seq_len: int
    n_layers: int
    d_model: int
    n_heads: int
    head_dim: int
    ffn_hidden_size: int
    norm_type: str
    norm_eps: float
    positional_encoding: str
    rope_theta: float
    activation: str
    attention_bias: bool
    mlp_bias: bool
    tie_word_embeddings: bool
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any]) -> "ModelConfig":
        allowed = {
            "name", "vocab_size", "max_seq_len", "n_layers", "d_model", "n_heads",
            "head_dim", "ffn_hidden_size", "norm_type", "norm_eps", "positional_encoding",
            "rope_theta", "activation", "attention_bias", "mlp_bias", "tie_word_embeddings",
            "embedding_dropout", "attention_dropout", "residual_dropout",
        }
        _check_keys(section, allowed, "model")
        required = {key: _required(section, key, "model") for key in (
            "name", "vocab_size", "max_seq_len", "n_layers", "d_model", "n_heads",
            "head_dim", "ffn_hidden_size", "norm_type", "norm_eps", "positional_encoding",
            "rope_theta", "activation", "attention_bias", "mlp_bias", "tie_word_embeddings",
        )}
        for key in ("vocab_size", "max_seq_len", "n_layers", "d_model", "n_heads", "head_dim", "ffn_hidden_size"):
            required[key] = _positive_int(required[key], f"model.{key}")
        for key in ("norm_eps", "rope_theta"):
            value = required[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"'model.{key}' must be a positive number")
            required[key] = float(value)
        if not isinstance(required["name"], str) or not required["name"].strip():
            raise ValueError("'model.name' must be a non-empty string")
        for key in ("norm_type", "positional_encoding", "activation"):
            if not isinstance(required[key], str) or not required[key].strip():
                raise ValueError(f"'model.{key}' must be a non-empty string")
        for key in ("attention_bias", "mlp_bias", "tie_word_embeddings"):
            if not isinstance(required[key], bool):
                raise ValueError(f"'model.{key}' must be a boolean")
        if required["d_model"] % required["n_heads"] != 0:
            raise ValueError("'model.d_model' must be divisible by 'model.n_heads'")
        expected = required["d_model"] // required["n_heads"]
        if required["head_dim"] != expected:
            raise ValueError("'model.head_dim' must equal model.d_model // model.n_heads")
        if required["head_dim"] % 2 != 0:
            raise ValueError("'model.head_dim' must be even for RoPE")
        if required["norm_type"].lower() != "rmsnorm":
            raise ValueError("'model.norm_type' must be rmsnorm")
        if required["positional_encoding"].lower() != "rope":
            raise ValueError("'model.positional_encoding' must be rope")
        if required["activation"].lower() != "swiglu":
            raise ValueError("'model.activation' must be swiglu")
        for key in ("embedding_dropout", "attention_dropout", "residual_dropout"):
            required[key] = _probability(section.get(key, 0.0), f"model.{key}")
        return cls(**required)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any]) -> "TrainingConfig":
        _check_keys(section, {"seed"}, "training")
        return cls(seed=_non_negative_int(_required(section, "seed", "training"), "training.seed"))


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    unk_token_id: int

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any]) -> "TokenizerConfig":
        fields = ("vocab_size", "pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id")
        _check_keys(section, set(fields), "tokenizer")
        values = {field: _positive_int(_required(section, field, "tokenizer"), f"tokenizer.{field}")
                  for field in ("vocab_size",)}
        for field in fields[1:]:
            values[field] = _non_negative_int(_required(section, field, "tokenizer"), f"tokenizer.{field}")
            if values[field] >= values["vocab_size"]:
                raise ValueError(f"'tokenizer.{field}' must be smaller than tokenizer.vocab_size")
        ids = [values[field] for field in fields[1:]]
        if len(set(ids)) != len(ids):
            raise ValueError("Tokenizer special token IDs must be unique")
        return cls(**values)


@dataclass(frozen=True)
class GenPyConfig:
    model: ModelConfig
    training: TrainingConfig
    tokenizer: TokenizerConfig


def load_config(path: PathLike) -> GenPyConfig:
    """Load and validate a complete GenPy YAML configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw = _mapping(raw, "configuration")
    expected_sections = {"model", "training", "tokenizer"}
    unknown = sorted(set(raw) - expected_sections)
    missing = sorted(expected_sections - set(raw))
    if unknown:
        raise ValueError(f"Unknown top-level configuration section(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"Missing required configuration section(s): {', '.join(missing)}")
    return GenPyConfig(
        model=ModelConfig.from_mapping(_mapping(raw["model"], "model")),
        training=TrainingConfig.from_mapping(_mapping(raw["training"], "training")),
        tokenizer=TokenizerConfig.from_mapping(_mapping(raw["tokenizer"], "tokenizer")),
    )
