"""Tests for the canonical configuration and its validation rules."""

from pathlib import Path

import pytest

from genpy.config import ModelConfig, TokenizerConfig, load_config


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "model_200m.yaml"


def test_canonical_config_loads() -> None:
    config = load_config(CONFIG_PATH)
    assert config.model.name == "GenPy-200M"
    assert config.model.vocab_size == 32000
    assert config.model.max_seq_len == 1024
    assert config.model.n_layers == 24
    assert config.model.d_model == 768
    assert config.model.n_heads == 12
    assert config.model.head_dim == 64
    assert config.model.ffn_hidden_size == 2176
    assert config.model.d_model // config.model.n_heads == config.model.head_dim


def test_invalid_attention_dimensions_are_rejected() -> None:
    values = {"name": "bad", "vocab_size": 10, "max_seq_len": 10, "n_layers": 1,
              "d_model": 10, "n_heads": 3, "head_dim": 3, "ffn_hidden_size": 10,
              "norm_type": "rmsnorm", "norm_eps": 1e-5, "positional_encoding": "rope",
              "rope_theta": 10000.0, "activation": "swiglu", "attention_bias": False,
              "mlp_bias": False, "tie_word_embeddings": True}
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig.from_mapping(values)


def test_invalid_dropout_is_rejected() -> None:
    values = {"name": "bad", "vocab_size": 10, "max_seq_len": 10, "n_layers": 1,
              "d_model": 8, "n_heads": 2, "head_dim": 4, "ffn_hidden_size": 10,
              "norm_type": "rmsnorm", "norm_eps": 1e-5, "positional_encoding": "rope",
              "rope_theta": 10000.0, "activation": "swiglu", "attention_bias": False,
              "mlp_bias": False, "tie_word_embeddings": True, "attention_dropout": 1.0}
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig.from_mapping(values)


def test_duplicate_special_token_ids_are_rejected() -> None:
    values = {"vocab_size": 10, "pad_token_id": 0, "bos_token_id": 0,
              "eos_token_id": 2, "unk_token_id": 3}
    with pytest.raises(ValueError, match="unique"):
        TokenizerConfig.from_mapping(values)
