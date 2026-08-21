"""Project-facing wrapper around a local GenPy tokenizer artifact."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer

from .config import SPECIAL_IDS, SPECIAL_TOKENS


class GenPyTokenizer:
    """A reversible local Byte-Level BPE tokenizer."""

    def __init__(self, backend: Tokenizer, name: str = "GenPy-Tokenizer-32K", expected_vocab_size: int = 32000) -> None:
        self._backend = backend
        self.name = name
        self.expected_vocab_size = expected_vocab_size
        self._special_by_id = {SPECIAL_IDS[key]: SPECIAL_TOKENS[key] for key in SPECIAL_IDS}
        self._special_ids = set(self._special_by_id)

    @classmethod
    def load(cls, directory: str | Path) -> "GenPyTokenizer":
        directory = Path(directory)
        artifact = directory / "tokenizer.json"
        if not artifact.is_file():
            raise FileNotFoundError(f"local tokenizer artifact not found: {artifact}")
        config_path = directory / "tokenizer_config.json"
        name = "GenPy-Tokenizer-32K"
        expected_vocab_size = 32000
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            name = config.get("name", name)
            expected_vocab_size = int(config.get("vocab_size", expected_vocab_size))
        tokenizer = cls(Tokenizer.from_file(str(artifact)), name=name, expected_vocab_size=expected_vocab_size)
        tokenizer._assert_contract()
        return tokenizer

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "GenPyTokenizer":
        """Compatibility alias for loading a local GenPy artifact only."""
        return cls.load(directory)

    @property
    def vocab_size(self) -> int:
        return self._backend.get_vocab_size(with_added_tokens=True)

    @property
    def pad_token_id(self) -> int:
        return SPECIAL_IDS["pad"]

    @property
    def bos_token_id(self) -> int:
        return SPECIAL_IDS["bos"]

    @property
    def eos_token_id(self) -> int:
        return SPECIAL_IDS["eos"]

    @property
    def unk_token_id(self) -> int:
        return SPECIAL_IDS["unk"]

    def _assert_contract(self) -> None:
        if self.vocab_size != self.expected_vocab_size:
            raise ValueError(f"expected {self.expected_vocab_size} vocabulary entries, got {self.vocab_size}")
        for key, token in SPECIAL_TOKENS.items():
            actual = self._backend.token_to_id(token)
            if actual != SPECIAL_IDS[key]:
                raise ValueError(f"{token} has ID {actual}, expected {SPECIAL_IDS[key]}")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self._backend.encode(text, add_special_tokens=True).ids)
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [list(encoded.ids) for encoded in self._backend.encode_batch(texts)]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        pieces: list[str] = []
        ordinary: list[int] = []

        def flush() -> None:
            if ordinary:
                pieces.append(self._backend.decode(ordinary, skip_special_tokens=False))
                ordinary.clear()

        for token_id in token_ids:
            if token_id in self._special_ids:
                flush()
                if not skip_special_tokens:
                    pieces.append(self._special_by_id[token_id])
            else:
                ordinary.append(int(token_id))
        flush()
        return "".join(pieces)

    def token_strings(self, token_ids: list[int]) -> list[str]:
        return [self._special_by_id[token_id] if token_id in self._special_ids else self._backend.id_to_token(token_id) or "<INVALID>" for token_id in token_ids]
