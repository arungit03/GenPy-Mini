from pathlib import Path

import pytest

from genpy.tokenizer.config import TokenizerConfig, load_tokenizer_config


def test_valid_config_and_contract() -> None:
    config = load_tokenizer_config(Path("configs/tokenizer.yaml"))
    assert config.vocab_size == 32000
    assert config.special_token_ids == {"pad": 0, "bos": 1, "eos": 2, "unk": 3}


def test_invalid_vocab_size_rejected() -> None:
    with pytest.raises(ValueError):
        TokenizerConfig(vocab_size=259).validate()


def test_duplicate_special_ids_rejected() -> None:
    config = TokenizerConfig()
    config.validate()
    from genpy.tokenizer import config as module
    original = module.SPECIAL_IDS.copy()
    try:
        module.SPECIAL_IDS["eos"] = 1
        with pytest.raises(ValueError):
            config.validate()
    finally:
        module.SPECIAL_IDS.clear()
        module.SPECIAL_IDS.update(original)
