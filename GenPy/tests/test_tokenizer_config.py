from pathlib import Path

import pytest
import yaml

from genpy.config import DataPipelineConfig, load_tokenizer_config, validate_tokenizer_vocab_contract


ROOT = Path(__file__).resolve().parents[1]


def test_load_tokenizer_config():
    config = load_tokenizer_config(ROOT / "configs" / "tokenizer.yaml")
    assert config.tokenizer.vocab_size == 32000
    assert config.tokenizer.special_tokens.ordered == ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>")
    assert config.tokenizer.add_prefix_space is False
    assert config.tokenizer.normalizer == "nfc"


def test_model_tokenizer_contract():
    config = load_tokenizer_config(ROOT / "configs" / "tokenizer.yaml")
    validate_tokenizer_vocab_contract(ROOT / "configs" / "model_200m.yaml", config)


def test_invalid_tokenizer_values():
    mapping = yaml.safe_load((ROOT / "configs" / "tokenizer.yaml").read_text(encoding="utf-8"))
    mapping["tokenizer"]["vocab_size"] = 512
    with pytest.raises(ValueError, match="32000"):
        from genpy.config import _build_tokenizer_config
        _build_tokenizer_config(mapping["tokenizer"])
    mapping["tokenizer"]["vocab_size"] = 32000
    mapping["tokenizer"]["min_frequency"] = 0
    path = ROOT / "tests" / "fixtures" / "invalid_tokenizer_config.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            load_tokenizer_config(path)
    finally:
        path.unlink()
