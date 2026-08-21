"""Configuration and validation for the GenPy tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SPECIAL_TOKENS = {"pad": "<PAD>", "bos": "<BOS>", "eos": "<EOS>", "unk": "<UNK>"}
SPECIAL_IDS = {"pad": 0, "bos": 1, "eos": 2, "unk": 3}


@dataclass(frozen=True)
class TokenizerConfig:
    name: str = "GenPy-Tokenizer-32K"
    tokenizer_type: str = "byte_level_bpe"
    vocab_size: int = 32000
    min_frequency: int = 2
    add_prefix_space: bool = False
    include_instruction: bool = True
    include_input: bool = True
    include_response: bool = True
    train_split_only: bool = True
    seed: int = 42
    require_roundtrip: bool = True
    maximum_unknown_rate: float = 0.0

    @property
    def special_tokens(self) -> dict[str, str]:
        return dict(SPECIAL_TOKENS)

    @property
    def special_token_ids(self) -> dict[str, int]:
        return dict(SPECIAL_IDS)

    def validate(self) -> None:
        if self.tokenizer_type != "byte_level_bpe":
            raise ValueError("tokenizer.type must be byte_level_bpe")
        if self.vocab_size < 260:
            raise ValueError("vocab_size must leave room for four specials and 256 byte tokens")
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        if self.maximum_unknown_rate < 0:
            raise ValueError("maximum_unknown_rate cannot be negative")
        if len(set(SPECIAL_IDS.values())) != len(SPECIAL_IDS):
            raise ValueError("special token IDs must be unique")


def load_tokenizer_config(path: str | Path) -> TokenizerConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    section = raw.get("tokenizer", raw)
    if not isinstance(section, dict):
        raise ValueError("tokenizer configuration must contain a mapping")
    specials = section.get("special_tokens", {})
    ids: dict[str, int] = {}
    tokens: dict[str, str] = {}
    for name in SPECIAL_TOKENS:
        item = specials.get(name, {})
        tokens[name] = str(item.get("token", SPECIAL_TOKENS[name]))
        ids[name] = int(item.get("id", SPECIAL_IDS[name]))
    if tokens != SPECIAL_TOKENS:
        raise ValueError(f"special token strings must be {SPECIAL_TOKENS}")
    if ids != SPECIAL_IDS or len(set(ids.values())) != len(ids):
        raise ValueError("special token IDs must be unique and exactly PAD=0, BOS=1, EOS=2, UNK=3")
    training = section.get("training", {}) or {}
    validation = section.get("validation", {}) or {}
    config = TokenizerConfig(
        name=str(section.get("name", "GenPy-Tokenizer-32K")),
        tokenizer_type=str(section.get("type", "byte_level_bpe")),
        vocab_size=int(section.get("vocab_size", 32000)),
        min_frequency=int(section.get("min_frequency", 2)),
        add_prefix_space=bool(section.get("add_prefix_space", False)),
        include_instruction=bool(training.get("include_instruction", True)),
        include_input=bool(training.get("include_input", True)),
        include_response=bool(training.get("include_response", True)),
        train_split_only=bool(training.get("train_split_only", True)),
        seed=int(training.get("seed", 42)),
        require_roundtrip=bool(validation.get("require_roundtrip", True)),
        maximum_unknown_rate=float(validation.get("maximum_unknown_rate", 0.0)),
    )
    config.validate()
    return config


def config_dict(config: TokenizerConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "type": config.tokenizer_type,
        "vocab_size": config.vocab_size,
        "min_frequency": config.min_frequency,
        "add_prefix_space": config.add_prefix_space,
        "special_tokens": {
            name: {"token": SPECIAL_TOKENS[name], "id": SPECIAL_IDS[name]}
            for name in SPECIAL_TOKENS
        },
        "training": {
            "include_instruction": config.include_instruction,
            "include_input": config.include_input,
            "include_response": config.include_response,
            "train_split_only": config.train_split_only,
            "seed": config.seed,
        },
        "validation": {
            "require_roundtrip": config.require_roundtrip,
            "maximum_unknown_rate": config.maximum_unknown_rate,
        },
    }
