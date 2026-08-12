"""Beginner-readable wrapper around the raw tokenizers.Tokenizer object."""

from pathlib import Path
from typing import Iterable, List, Optional

from tokenizers import Tokenizer


class GenPyTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: Path) -> "GenPyTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    @property
    def raw(self) -> Tokenizer:
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def _id(self, token: str) -> int:
        value = self._tokenizer.token_to_id(token)
        if value is None:
            raise ValueError(f"Tokenizer does not contain required token: {token}")
        return value

    @property
    def pad_token_id(self) -> int:
        return self._id("<|pad|>")

    @property
    def bos_token_id(self) -> int:
        return self._id("<|bos|>")

    @property
    def eos_token_id(self) -> int:
        return self._id("<|eos|>")

    @property
    def unk_token_id(self) -> int:
        return self._id("<|unk|>")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        ids = list(encoding.ids)
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def token_strings(self, ids: Iterable[int]) -> List[str]:
        return [self._tokenizer.id_to_token(int(token_id)) or "<missing>" for token_id in ids]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def encode_document(self, text: str) -> List[int]:
        """Encode a document and append EOS, without adding BOS or packing."""
        return self.encode(text, add_eos=True)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))
