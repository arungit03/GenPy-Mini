"""Stable GenPy tokenizer wrapper for future model phases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from genpy.tokenizer.config import SPECIAL_TOKEN_NAMES, SPECIAL_TOKEN_TEXT
from genpy.tokenizer.fingerprint import read_checksum_file, sha256_file
from genpy.tokenizer.serialization import (
    serialize_instruction,
    serialize_pretraining,
    validate_content,
)


class TokenizerArtifactError(RuntimeError):
    """Raised for corrupted or incompatible tokenizer artifacts."""


@dataclass(frozen=True, slots=True)
class TokenizerValidationResult:
    """High-level wrapper validation result."""

    passed: bool
    errors: tuple[str, ...]
    actual_vocab_size: int
    fingerprint: str


class GenPyTokenizer:
    """Implementation-independent API around GenPy's custom tokenizer."""

    def __init__(
        self, tokenizer: Tokenizer, metadata: dict[str, object], artifact_path: Path
    ) -> None:
        self._tokenizer = tokenizer
        self._metadata = metadata
        self._artifact_path = artifact_path
        raw_special = metadata.get("special_tokens")
        if not isinstance(raw_special, dict):
            raise TokenizerArtifactError("metadata is missing special_tokens")
        special_ids: dict[str, int] = {}
        for name in SPECIAL_TOKEN_NAMES:
            item = raw_special.get(name)
            if not isinstance(item, dict) or not isinstance(item.get("id"), int):
                raise TokenizerArtifactError(f"metadata special token is invalid: {name}")
            special_ids[name] = item["id"]
        self.special_token_ids = special_ids

    @property
    def fingerprint(self) -> str:
        """Return the immutable artifact identity."""
        value = self._metadata.get("tokenizer_fingerprint")
        if not isinstance(value, str) or len(value) != 64:
            raise TokenizerArtifactError("metadata has an invalid tokenizer fingerprint")
        return value

    @property
    def vocab_size(self) -> int:
        """Return the complete vocabulary size including special tokens."""
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def encode_text(self, text: str) -> list[int]:
        """Encode ordinary text after enforcing the control-token collision policy."""
        validate_content(text, "text")
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode IDs after strict range validation; decoded text is never executed."""
        for token_id in ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"token ID outside vocabulary: {token_id}")
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_pretraining_record(self, code: str) -> list[int]:
        """Serialize and encode one pretraining sample without truncation."""
        serialized = serialize_pretraining(code)
        return list(self._tokenizer.encode(serialized.text, add_special_tokens=False).ids)

    def encode_instruction_record(self, prompt: str, code: str) -> list[int]:
        """Serialize and encode one instruction sample without truncation."""
        serialized = serialize_instruction(prompt, code)
        return list(self._tokenizer.encode(serialized.text, add_special_tokens=False).ids)

    def token_to_id(self, token: str) -> int:
        """Resolve a token or fail instead of returning an unknown fallback."""
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"token is not in vocabulary: {token!r}")
        return token_id

    def id_to_token(self, token_id: int) -> str:
        """Resolve a validated token ID."""
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"token ID outside vocabulary: {token_id}")
        token = self._tokenizer.id_to_token(token_id)
        if token is None:
            raise TokenizerArtifactError(f"vocabulary has no token for ID {token_id}")
        return token

    def sequence_length(self, ids: list[int]) -> int:
        """Return sequence length separately from encoding and never truncate."""
        return len(ids)

    def save(self, path: Path) -> None:
        """Atomically save tokenizer JSON without altering the source artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        self._tokenizer.save(str(temporary))
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> GenPyTokenizer:
        """Load only after validating required files and all recorded checksums."""
        required = {
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "metadata.json",
            "corpus_fingerprint.json",
            "evaluation.json",
            "checksums.sha256",
        }
        missing = sorted(name for name in required if not (path / name).is_file())
        if missing:
            raise TokenizerArtifactError(f"artifact is missing files: {', '.join(missing)}")
        try:
            checksums = read_checksum_file(path / "checksums.sha256")
            for filename, expected in checksums.items():
                if filename == "checksums.sha256" or sha256_file(path / filename) != expected:
                    raise TokenizerArtifactError(f"artifact checksum failed: {filename}")
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise TokenizerArtifactError("metadata must be an object")
            tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            if isinstance(error, TokenizerArtifactError):
                raise
            raise TokenizerArtifactError("tokenizer artifact could not be loaded") from error
        wrapper = cls(tokenizer, metadata, path)
        validation = wrapper.validate()
        if not validation.passed:
            raise TokenizerArtifactError(
                "tokenizer validation failed: " + "; ".join(validation.errors)
            )
        return wrapper

    def validate(self) -> TokenizerValidationResult:
        """Validate vocabulary size, special IDs, atomicity, and representative round trips."""
        errors: list[str] = []
        raw_expected_vocab = self._metadata.get("actual_vocabulary_size")
        if not isinstance(raw_expected_vocab, int):
            errors.append("metadata vocabulary size is invalid")
            expected_vocab = -1
        else:
            expected_vocab = raw_expected_vocab
        if self.vocab_size != expected_vocab:
            errors.append("vocabulary size differs from metadata")
        for expected_id, (name, token) in enumerate(
            zip(SPECIAL_TOKEN_NAMES, SPECIAL_TOKEN_TEXT, strict=True)
        ):
            if self.special_token_ids.get(name) != expected_id:
                errors.append(f"metadata special-token ID differs: {name}")
            encoding = self._tokenizer.encode(token, add_special_tokens=False).ids
            if encoding != [expected_id]:
                errors.append(f"special token is not atomic: {name}")
        examples = (
            "def add(a, b):\n    return a + b\n",
            "if True:\n\tprint('tab')\n",
            "தமிழ் 😀 café\n",
            'text = """line one\nline two"""\n',
        )
        for example in examples:
            ids = self._tokenizer.encode(example, add_special_tokens=False).ids
            if self._tokenizer.decode(ids, skip_special_tokens=False) != example:
                errors.append("UTF-8 round-trip failed")
                break
        return TokenizerValidationResult(
            not errors, tuple(errors), self.vocab_size, self.fingerprint
        )
