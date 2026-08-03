from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    assert isinstance(data, dict)
    return data


def test_genpy_package_can_be_imported() -> None:
    import genpy

    assert genpy.PROJECT_NAME == "GenPy"


def test_yaml_configuration_files_can_be_loaded() -> None:
    config_paths = [
        PROJECT_ROOT / "configs" / "model" / "genpy_5m.yaml",
        PROJECT_ROOT / "configs" / "model" / "genpy_25m.yaml",
        PROJECT_ROOT / "configs" / "model" / "genpy_100m.yaml",
        PROJECT_ROOT / "configs" / "pretrain.yaml",
        PROJECT_ROOT / "configs" / "instruction_train.yaml",
        PROJECT_ROOT / "configs" / "evaluate.yaml",
    ]

    for path in config_paths:
        assert path.is_file(), f"Missing config file: {path}"
        assert load_yaml(path)


def test_required_folders_exist() -> None:
    required_folders = [
        "configs/model",
        "data/raw",
        "data/cleaned",
        "data/tokenized",
        "data/instruction",
        "data/evaluation",
        "docs",
        "notebooks/kaggle",
        "scripts",
        "src/genpy/config",
        "src/genpy/data",
        "src/genpy/tokenizer",
        "src/genpy/model",
        "src/genpy/training",
        "src/genpy/evaluation",
        "src/genpy/inference",
        "src/genpy/utils",
        "tests/unit",
        "tests/integration",
        "artifacts",
        "checkpoints",
        "logs",
    ]

    for folder in required_folders:
        assert (PROJECT_ROOT / folder).is_dir(), f"Missing folder: {folder}"


def test_100m_configuration_contains_required_fields() -> None:
    config = load_yaml(PROJECT_ROOT / "configs" / "model" / "genpy_100m.yaml")
    model = config["model"]

    required_fields = {
        "name",
        "architecture",
        "vocab_size",
        "context_length",
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "head_dimension",
        "intermediate_size",
        "activation",
        "normalization",
        "positional_encoding",
        "tie_word_embeddings",
        "attention_dropout",
        "residual_dropout",
        "initialization",
    }

    assert required_fields.issubset(model)
    assert model["name"] == "GenPy-100M"
    assert model["initialization"] == "random"
    assert model["vocab_size"] == 16384


def test_all_model_configs_share_tokenizer_contract() -> None:
    expected_ids = {
        "pad": 0,
        "bos": 1,
        "eos": 2,
        "user": 3,
        "assistant": 4,
        "code": 5,
        "end": 6,
    }

    for path in sorted((PROJECT_ROOT / "configs" / "model").glob("genpy_*.yaml")):
        config = load_yaml(path)
        model = config["model"]
        tokenizer = config["tokenizer"]

        assert model["architecture"] == "decoder_only_transformer"
        assert model["attention"] == "causal_self_attention"
        assert model["initialization"] == "random"
        assert model["vocab_size"] == 16384
        assert model["context_length"] == 1024
        assert tokenizer["vocab_size"] == 16384
        assert tokenizer["trained"] is False
        assert tokenizer["special_token_ids"] == expected_ids
