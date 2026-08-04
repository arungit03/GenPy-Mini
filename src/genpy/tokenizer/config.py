"""Typed Phase 3 tokenizer configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

SPECIAL_TOKEN_NAMES = ("pad", "bos", "eos", "user", "assistant", "code", "end")
SPECIAL_TOKEN_TEXT = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|user|>",
    "<|assistant|>",
    "<|code|>",
    "<|end|>",
)


class TokenizerConfigError(ValueError):
    """Raised when a tokenizer configuration violates a locked project decision."""


@dataclass(frozen=True, slots=True)
class SpecialToken:
    """One locked control token and ID."""

    token: str
    id: int


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """Validated tokenizer configuration and its repository root."""

    path: Path
    project_root: Path
    raw: dict[str, Any]
    config_hash: str
    special_tokens: dict[str, SpecialToken]

    @property
    def tokenizer(self) -> dict[str, Any]:
        """Return tokenizer settings."""
        return cast(dict[str, Any], self.raw["tokenizer"])

    @property
    def corpus(self) -> dict[str, Any]:
        """Return corpus settings."""
        return cast(dict[str, Any], self.raw["corpus"])

    def resolve(self, value: str) -> Path:
        """Resolve a configured path without hard-coded machine locations."""
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    @property
    def artifact_path(self) -> Path:
        """Return the configured artifact directory."""
        return self.resolve(str(self.tokenizer["artifact_path"]))

    @property
    def ordered_special_tokens(self) -> tuple[str, ...]:
        """Return control-token text in permanent ID order."""
        return tuple(self.special_tokens[name].token for name in SPECIAL_TOKEN_NAMES)


def _find_project_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def load_tokenizer_config(path: Path, project_root: Path | None = None) -> TokenizerConfig:
    """Load YAML and reject settings that conflict with Phase 3 decisions."""
    resolved = path.resolve()
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TokenizerConfigError("tokenizer config must be a mapping")
    for section in ("tokenizer", "special_tokens", "corpus", "readiness", "runtime"):
        if not isinstance(value.get(section), dict):
            raise TokenizerConfigError(f"missing mapping: {section}")
    settings = value["tokenizer"]
    if settings.get("algorithm") != "byte_level_bpe":
        raise TokenizerConfigError("algorithm must be byte_level_bpe")
    if settings.get("status") == "production" and int(settings.get("vocab_size", 0)) != 16384:
        raise TokenizerConfigError("production vocabulary size must be exactly 16384")
    if int(settings.get("vocab_size", 0)) != int(settings.get("expected_vocab_size", -1)):
        raise TokenizerConfigError("vocab_size and expected_vocab_size differ")
    if settings.get("unicode_normalization") != "none" or settings.get("lowercase") is not False:
        raise TokenizerConfigError("normalization and lowercasing must remain disabled")
    if float(settings.get("dropout", -1)) != 0.0:
        raise TokenizerConfigError("deterministic tokenizer dropout must be 0.0")
    raw_special = value["special_tokens"]
    special: dict[str, SpecialToken] = {}
    for expected_id, (name, token_text) in enumerate(
        zip(SPECIAL_TOKEN_NAMES, SPECIAL_TOKEN_TEXT, strict=True)
    ):
        item = raw_special.get(name)
        if not isinstance(item, dict):
            raise TokenizerConfigError(f"missing special token: {name}")
        token = SpecialToken(token=str(item.get("token")), id=int(item.get("id", -1)))
        if token.token != token_text or token.id != expected_id:
            raise TokenizerConfigError(
                f"special token {name} must be {token_text!r} at {expected_id}"
            )
        special[name] = token
    mixture = value["corpus"].get("mixture", {})
    total_weight = float(mixture.get("pretraining_weight", 0)) + float(
        mixture.get("instruction_weight", 0)
    )
    if abs(total_weight - 1.0) > 1e-9:
        raise TokenizerConfigError("corpus mixture weights must sum to 1.0")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return TokenizerConfig(
        path=resolved,
        project_root=(project_root or _find_project_root(resolved)).resolve(),
        raw=value,
        config_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        special_tokens=special,
    )


def validate_model_tokenizer_contracts(project_root: Path) -> dict[str, Any]:
    """Validate the shared production-tokenizer contract across every model scale."""
    paths = sorted((project_root / "configs/model").glob("genpy_*.yaml"))
    if not paths:
        raise TokenizerConfigError("no GenPy model configurations were found")
    expected_ids = dict(zip(SPECIAL_TOKEN_NAMES, range(len(SPECIAL_TOKEN_NAMES)), strict=True))
    contract_keys = (
        "name",
        "version",
        "type",
        "vocab_size",
        "trained",
        "artifact_path",
        "fingerprint",
        "special_token_ids",
    )
    reference: dict[str, Any] | None = None
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
            raise TokenizerConfigError(f"invalid model configuration: {path.name}")
        tokenizer = raw.get("tokenizer")
        if not isinstance(tokenizer, dict):
            raise TokenizerConfigError(f"missing tokenizer contract: {path.name}")
        contract = {key: tokenizer.get(key) for key in contract_keys}
        if int(raw["model"].get("vocab_size", -1)) != 16384:
            raise TokenizerConfigError(f"model vocabulary size differs: {path.name}")
        if (
            tokenizer.get("vocab_size") != 16384
            or tokenizer.get("special_token_ids") != expected_ids
        ):
            raise TokenizerConfigError(f"tokenizer vocabulary contract differs: {path.name}")
        if reference is None:
            reference = contract
        elif contract != reference:
            raise TokenizerConfigError(f"model tokenizer contracts differ: {path.name}")
    assert reference is not None
    fingerprint = reference["fingerprint"]
    if fingerprint == "populated_after_training":
        if reference["trained"] is not False:
            raise TokenizerConfigError("an unfrozen tokenizer must have trained: false")
        return reference
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or reference["trained"] is not True
    ):
        raise TokenizerConfigError("frozen tokenizer fingerprint metadata is invalid")
    from genpy.tokenizer.tokenizer import GenPyTokenizer

    artifact = project_root / str(reference["artifact_path"])
    loaded = GenPyTokenizer.load(artifact)
    if loaded.vocab_size != 16384 or loaded.fingerprint != fingerprint:
        raise TokenizerConfigError("model tokenizer artifact does not match its contract")
    return reference
