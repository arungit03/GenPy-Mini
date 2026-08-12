"""Explicit validation of GenPy tokenizer artifacts and contracts."""

import json
from pathlib import Path
from typing import Optional

from tokenizers import Tokenizer

from genpy.config import PathLike, load_model_config

from .tokenizer import GenPyTokenizer
from .trainer import sha256_file


class TokenizerValidationError(ValueError):
    pass


def validate_tokenizer_artifact(
    tokenizer_path: PathLike,
    manifest_path: Optional[PathLike] = None,
    expected_vocab_size: int = 32000,
    model_config_path: Optional[PathLike] = None,
) -> dict:
    path = Path(tokenizer_path)
    if not path.is_file():
        raise TokenizerValidationError(f"Tokenizer file does not exist: {path}")
    try:
        raw = Tokenizer.from_file(str(path))
    except Exception as exc:
        raise TokenizerValidationError(f"Tokenizer JSON could not be loaded: {exc}") from exc
    wrapper = GenPyTokenizer(raw)
    if wrapper.vocab_size != expected_vocab_size:
        raise TokenizerValidationError(f"Vocabulary size mismatch: expected {expected_vocab_size}, got {wrapper.vocab_size}")
    expected_ids = {"pad": 0, "bos": 1, "eos": 2, "unk": 3}
    actual_ids = {"pad": wrapper.pad_token_id, "bos": wrapper.bos_token_id, "eos": wrapper.eos_token_id, "unk": wrapper.unk_token_id}
    if actual_ids != expected_ids:
        raise TokenizerValidationError(f"Special token IDs are incorrect: {actual_ids}")
    serialized = json.loads(raw.to_str())
    if serialized.get("normalizer", {}).get("type") != "NFC":
        raise TokenizerValidationError("Tokenizer normalizer is not NFC")
    pre = serialized.get("pre_tokenizer", {})
    if pre.get("type") != "ByteLevel" or pre.get("add_prefix_space") is not False or pre.get("use_regex") is not True:
        raise TokenizerValidationError("Tokenizer ByteLevel settings are incorrect")
    if serialized.get("decoder", {}).get("type") != "ByteLevel":
        raise TokenizerValidationError("Tokenizer decoder is not ByteLevel")
    if model_config_path is not None:
        model = load_model_config(model_config_path)
        if model.vocab_size != wrapper.vocab_size:
            raise TokenizerValidationError(f"Architecture vocabulary mismatch: model={model.vocab_size}, tokenizer={wrapper.vocab_size}")
    sample = "GenPy café தமிழ் हिन्दी 中文 🚀\ncode"
    ids = wrapper.encode(sample)
    if not ids or wrapper.decode(ids) != sample:
        raise TokenizerValidationError("Tokenizer encode/decode round-trip failed")
    result = {"tokenizer": "GenPy-Tokenizer", "vocab_size": wrapper.vocab_size, "special_token_ids": actual_ids, "round_trip": True, "checksum": None}
    if manifest_path is not None:
        manifest = Path(manifest_path)
        if not manifest.is_file():
            raise TokenizerValidationError(f"Tokenizer manifest does not exist: {manifest}")
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TokenizerValidationError(f"Tokenizer manifest is invalid: {exc}") from exc
        actual_checksum = sha256_file(path)
        if data.get("tokenizer_json_sha256") != actual_checksum:
            raise TokenizerValidationError("Tokenizer checksum does not match manifest")
        result["checksum"] = actual_checksum
    return result


def verify_checksum(tokenizer_path: PathLike, manifest_path: PathLike) -> bool:
    validate_tokenizer_artifact(tokenizer_path, manifest_path, expected_vocab_size=32000)
    return True
