from pathlib import Path

import pytest

from genpy.config import ModelConfig, TrainingConfig, load_model_config, load_training_config


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "configs" / "model_200m.yaml"
TRAIN_PATH = ROOT / "configs" / "train.yaml"


def test_load_model_configuration():
    config = load_model_config(MODEL_PATH)
    assert isinstance(config, ModelConfig)
    assert config.name == "GenPy-200M"
    assert config.vocab_size == 32000
    assert config.max_seq_len == 1024
    assert config.hidden_size == 768
    assert config.num_layers == 24
    assert config.num_heads == 12
    assert config.intermediate_size == 2176


def test_attention_dimensions():
    config = load_model_config(MODEL_PATH)
    assert config.hidden_size % config.num_heads == 0
    assert config.head_dim == config.hidden_size // config.num_heads == 64


def test_load_training_configuration():
    config = load_training_config(TRAIN_PATH)
    assert isinstance(config, TrainingConfig)
    assert config.seed == 42
    assert config.micro_batch_size == 4
    assert config.gradient_accumulation_steps == 16
    assert config.precision == "auto"


def _model_mapping(**overrides):
    base = {
        "name": "test", "vocab_size": 100, "max_seq_len": 8, "hidden_size": 8,
        "num_layers": 2, "num_heads": 2, "head_dim": 4, "intermediate_size": 16,
        "norm_eps": 1e-5, "rope_theta": 10000.0, "tie_embeddings": True,
    }
    base.update(overrides)
    return base


def test_invalid_attention_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig.from_mapping(_model_mapping(hidden_size=7))
    with pytest.raises(ValueError, match="head_dim"):
        ModelConfig.from_mapping(_model_mapping(head_dim=5))


@pytest.mark.parametrize("field", ["vocab_size", "max_seq_len", "hidden_size", "num_layers", "num_heads", "head_dim", "intermediate_size", "norm_eps", "rope_theta"])
def test_invalid_non_positive_model_values(field):
    with pytest.raises(ValueError, match="positive"):
        ModelConfig.from_mapping(_model_mapping(**{field: 0}))


def test_missing_configuration_sections(tmp_path):
    missing_model = tmp_path / "missing_model.yaml"
    missing_model.write_text("training: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model"):
        load_model_config(missing_model)
    missing_training = tmp_path / "missing_training.yaml"
    missing_training.write_text("model: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training"):
        load_training_config(missing_training)
