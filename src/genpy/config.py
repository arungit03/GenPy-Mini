"""Typed, immutable configuration objects for the GenPy-Mini architecture.

This module only loads and validates configuration data. It does not build,
import, or reference any tokenizer or model implementation -- those belong to
later phases.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Final

import yaml

_REQUIRED_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "seed",
    "model_name",
    "architecture",
    "task",
    "language_scope",
    "vocab_size",
    "context_length",
    "d_model",
    "num_layers",
    "num_attention_heads",
    "feed_forward_dimension",
    "dropout",
    "tie_input_output_embeddings",
    "target_parameter_range",
    "special_tokens",
)

_REQUIRED_SPECIAL_TOKEN_FIELDS: Final[tuple[str, ...]] = (
    "pad_token",
    "bos_token",
    "eos_token",
    "unk_token",
    "instruction_token",
    "input_token",
    "response_token",
    "code_token",
    "explanation_token",
)


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed, or inconsistent."""


@dataclasses.dataclass(frozen=True, slots=True)
class SpecialTokens:
    """Immutable collection of reserved vocabulary tokens."""

    pad_token: str
    bos_token: str
    eos_token: str
    unk_token: str
    instruction_token: str
    input_token: str
    response_token: str
    code_token: str
    explanation_token: str

    def values(self) -> tuple[str, ...]:
        """Return every token value, in field-declaration order."""
        return tuple(getattr(self, field.name) for field in dataclasses.fields(self))


@dataclasses.dataclass(frozen=True, slots=True)
class ModelConfig:
    """Immutable, validated GenPy-Mini architecture configuration."""

    schema_version: int
    seed: int
    model_name: str
    architecture: str
    task: str
    language_scope: tuple[str, ...]
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_attention_heads: int
    feed_forward_dimension: int
    dropout: float
    tie_input_output_embeddings: bool
    target_parameter_range_min: int
    target_parameter_range_max: int
    special_tokens: SpecialTokens


def load_model_config(path: Path) -> ModelConfig:
    """Load, validate, and return the GenPy-Mini configuration at ``path``.

    Raises:
        ConfigError: if the file is missing, is not valid YAML, does not
            contain a mapping, is missing a required field, or fails an
            architecture consistency check.
    """
    data = _read_yaml_mapping(path)
    _require_fields(data, _REQUIRED_TOP_LEVEL_FIELDS, source=path)

    special_tokens = _build_special_tokens(data["special_tokens"], source=path)
    range_min, range_max = _parse_parameter_range(data["target_parameter_range"], source=path)

    config = ModelConfig(
        schema_version=_as_int(data["schema_version"], "schema_version", source=path),
        seed=_as_int(data["seed"], "seed", source=path),
        model_name=str(data["model_name"]),
        architecture=str(data["architecture"]),
        task=str(data["task"]),
        language_scope=tuple(str(item) for item in data["language_scope"]),
        vocab_size=_as_int(data["vocab_size"], "vocab_size", source=path),
        context_length=_as_int(data["context_length"], "context_length", source=path),
        d_model=_as_int(data["d_model"], "d_model", source=path),
        num_layers=_as_int(data["num_layers"], "num_layers", source=path),
        num_attention_heads=_as_int(
            data["num_attention_heads"], "num_attention_heads", source=path
        ),
        feed_forward_dimension=_as_int(
            data["feed_forward_dimension"], "feed_forward_dimension", source=path
        ),
        dropout=_as_float(data["dropout"], "dropout", source=path),
        tie_input_output_embeddings=bool(data["tie_input_output_embeddings"]),
        target_parameter_range_min=range_min,
        target_parameter_range_max=range_max,
        special_tokens=special_tokens,
    )

    _validate_consistency(config, source=path)
    return config


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read configuration file {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Configuration file {path} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {path} must contain a mapping at the top level.")

    return data


def _require_fields(data: dict[str, Any], fields: tuple[str, ...], *, source: Path) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        raise ConfigError(
            f"Configuration file {source} is missing required field(s): {', '.join(missing)}"
        )


def _build_special_tokens(raw: Any, *, source: Path) -> SpecialTokens:
    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration file {source}: 'special_tokens' must be a mapping.")

    _require_fields(raw, _REQUIRED_SPECIAL_TOKEN_FIELDS, source=source)

    tokens = SpecialTokens(**{field: str(raw[field]) for field in _REQUIRED_SPECIAL_TOKEN_FIELDS})

    values = tokens.values()
    if len(set(values)) != len(values):
        raise ConfigError(
            f"Configuration file {source}: special_tokens values must be unique, got {values!r}."
        )

    return tokens


def _parse_parameter_range(raw: Any, *, source: Path) -> tuple[int, int]:
    text = str(raw)
    parts = text.split("-")
    if len(parts) != 2:
        raise ConfigError(
            f"Configuration file {source}: 'target_parameter_range' must look like "
            f"'<min>-<max>', got {text!r}."
        )

    try:
        range_min, range_max = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ConfigError(
            f"Configuration file {source}: 'target_parameter_range' bounds must be integers, "
            f"got {text!r}."
        ) from exc

    if range_min > range_max:
        raise ConfigError(
            f"Configuration file {source}: target_parameter_range minimum ({range_min}) "
            f"must not exceed maximum ({range_max})."
        )

    return range_min, range_max


def _as_int(value: Any, field: str, *, source: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"Configuration file {source}: '{field}' must be an integer, got {value!r}."
        )
    return int(value)


def _as_float(value: Any, field: str, *, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"Configuration file {source}: '{field}' must be a number, got {value!r}."
        )
    return float(value)


def _validate_consistency(config: ModelConfig, *, source: Path) -> None:
    if config.num_attention_heads <= 0:
        raise ConfigError(
            f"Configuration file {source}: num_attention_heads must be positive, "
            f"got {config.num_attention_heads}."
        )

    if config.d_model % config.num_attention_heads != 0:
        raise ConfigError(
            f"Configuration file {source}: d_model ({config.d_model}) must be divisible by "
            f"num_attention_heads ({config.num_attention_heads})."
        )

    if config.vocab_size <= len(config.special_tokens.values()):
        raise ConfigError(
            f"Configuration file {source}: vocab_size ({config.vocab_size}) must be greater than "
            f"the number of special tokens ({len(config.special_tokens.values())})."
        )

    if config.context_length <= 0:
        raise ConfigError(
            f"Configuration file {source}: context_length must be positive, "
            f"got {config.context_length}."
        )

    if config.num_layers <= 0:
        raise ConfigError(
            f"Configuration file {source}: num_layers must be positive, got {config.num_layers}."
        )

    if not (0.0 <= config.dropout <= 1.0):
        raise ConfigError(
            f"Configuration file {source}: dropout must be between 0.0 and 1.0, "
            f"got {config.dropout}."
        )

    if config.feed_forward_dimension < config.d_model:
        raise ConfigError(
            f"Configuration file {source}: feed_forward_dimension "
            f"({config.feed_forward_dimension}) must be greater than or equal to "
            f"d_model ({config.d_model})."
        )

    if config.target_parameter_range_min > config.target_parameter_range_max:
        raise ConfigError(
            f"Configuration file {source}: target_parameter_range minimum "
            f"({config.target_parameter_range_min}) must not exceed maximum "
            f"({config.target_parameter_range_max})."
        )
