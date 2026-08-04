from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from genpy.tokenizer.config import (
    TokenizerConfigError,
    load_tokenizer_config,
    validate_model_tokenizer_contracts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_production_config_locks_vocabulary_and_special_ids() -> None:
    config = load_tokenizer_config(Path("configs/tokenizer/genpy_bpe_16k.yaml"))
    assert config.tokenizer["vocab_size"] == 16384
    assert [config.special_tokens[name].id for name in config.special_tokens] == list(range(7))


def test_invalid_production_vocabulary_is_rejected(tmp_path: Path) -> None:
    value = yaml.safe_load(Path("configs/tokenizer/genpy_bpe_16k.yaml").read_text())
    value["tokenizer"]["vocab_size"] = 1024
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(TokenizerConfigError, match="16384"):
        load_tokenizer_config(path)


def test_model_configs_share_unfrozen_production_contract() -> None:
    contract = validate_model_tokenizer_contracts(PROJECT_ROOT)
    assert contract["fingerprint"] == "populated_after_training"
    assert contract["vocab_size"] == 16384
