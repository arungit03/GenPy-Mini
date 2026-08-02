"""Tests for the GenPy-Mini model configuration loader and validation rules."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

from config.settings import MODEL_CONFIG_PATH
from genpy.config import ConfigError, ModelConfig, load_model_config


def _base_config_dict() -> dict[str, Any]:
    """Return a valid GenPy-Mini configuration mapping to mutate per test."""
    return {
        "schema_version": 1,
        "seed": 42,
        "model_name": "GenPy-Mini",
        "architecture": "decoder_only_transformer",
        "task": "causal_language_modeling",
        "language_scope": ["Python", "English programming instructions"],
        "vocab_size": 16000,
        "context_length": 512,
        "d_model": 512,
        "num_layers": 8,
        "num_attention_heads": 8,
        "feed_forward_dimension": 2048,
        "dropout": 0.1,
        "tie_input_output_embeddings": True,
        "target_parameter_range": "33000000-35000000",
        "special_tokens": {
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "instruction_token": "<instruction>",
            "input_token": "<input>",
            "response_token": "<response>",
            "code_token": "<code>",
            "explanation_token": "<explanation>",
        },
    }


def _write_config(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "model_config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_valid_configuration_loads() -> None:
    config = load_model_config(MODEL_CONFIG_PATH)
    assert isinstance(config, ModelConfig)


def test_architecture_matches_expected_genpy_mini_values() -> None:
    config = load_model_config(MODEL_CONFIG_PATH)
    assert config.model_name == "GenPy-Mini"
    assert config.architecture == "decoder_only_transformer"
    assert config.task == "causal_language_modeling"
    assert config.vocab_size == 16000
    assert config.context_length == 512
    assert config.d_model == 512
    assert config.num_layers == 8
    assert config.num_attention_heads == 8
    assert config.feed_forward_dimension == 2048
    assert config.dropout == pytest.approx(0.1)
    assert config.tie_input_output_embeddings is True
    assert config.target_parameter_range_min == 33_000_000
    assert config.target_parameter_range_max == 35_000_000


def test_d_model_not_divisible_by_heads_is_rejected(tmp_path: Path) -> None:
    data = _base_config_dict()
    data["d_model"] = 500
    data["num_attention_heads"] = 8
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="d_model"):
        load_model_config(path)


def test_dropout_out_of_range_is_rejected(tmp_path: Path) -> None:
    data = _base_config_dict()
    data["dropout"] = 1.5
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="dropout"):
        load_model_config(path)


def test_duplicate_special_tokens_are_rejected(tmp_path: Path) -> None:
    data = _base_config_dict()
    data["special_tokens"]["unk_token"] = data["special_tokens"]["pad_token"]
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="special_tokens"):
        load_model_config(path)


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    data = _base_config_dict()
    del data["vocab_size"]
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="vocab_size"):
        load_model_config(path)


def test_invalid_parameter_range_is_rejected(tmp_path: Path) -> None:
    data = _base_config_dict()
    data["target_parameter_range"] = "35000000-33000000"
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="target_parameter_range"):
        load_model_config(path)


def test_configuration_object_is_immutable() -> None:
    config = load_model_config(MODEL_CONFIG_PATH)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.vocab_size = 1  # type: ignore[misc]


def test_missing_configuration_file_raises_useful_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist.yaml"

    with pytest.raises(ConfigError, match="not found"):
        load_model_config(missing_path)


def test_malformed_yaml_raises_useful_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("d_model: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigError, match="YAML"):
        load_model_config(path)


def test_vocab_size_smaller_than_special_tokens_is_rejected(tmp_path: Path) -> None:
    data = _base_config_dict()
    data["vocab_size"] = 5
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="vocab_size"):
        load_model_config(path)


def test_feed_forward_dimension_below_d_model_is_rejected(tmp_path: Path) -> None:
    data = _base_config_dict()
    data["feed_forward_dimension"] = 100
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="feed_forward_dimension"):
        load_model_config(path)
