"""Binary packed-shard format and metadata helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from genpy.tokenizer.fingerprint import atomic_write_json, sha256_file

TOKEN_DTYPE = np.dtype("<u2")
MASK_DTYPE = np.dtype("u1")


@dataclass(frozen=True, slots=True)
class PackedShard:
    """Validated metadata needed to memory-map one packed shard."""

    metadata_path: Path
    tokens_path: Path
    loss_mask_path: Path
    sample_count: int
    stored_token_width: int
    context_length: int
    tokenizer_fingerprint: str
    family: str
    split: str


def write_binary(path: Path, values: np.ndarray) -> None:
    """Atomically write a contiguous NumPy array without a pickle header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(np.ascontiguousarray(values).tobytes(order="C"))
    temporary.replace(path)


def load_shard_metadata(path: Path) -> tuple[dict[str, Any], PackedShard]:
    """Load metadata and resolve its relative binary filenames."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"packed metadata cannot be loaded: {path.name}") from error
    if not isinstance(raw, dict):
        raise ValueError("packed metadata must be an object")
    shard = PackedShard(
        metadata_path=path,
        tokens_path=path.parent / str(raw["tokens_file"]),
        loss_mask_path=path.parent / str(raw["loss_mask_file"]),
        sample_count=int(raw["sample_count"]),
        stored_token_width=int(raw["stored_token_width"]),
        context_length=int(raw["input_sequence_length"]),
        tokenizer_fingerprint=str(raw["tokenizer_fingerprint"]),
        family=str(raw["family"]),
        split=str(raw["split"]),
    )
    return raw, shard


def validate_shard(path: Path, expected_fingerprint: str, expected_config_hash: str) -> PackedShard:
    """Validate checksums, sizes, dtypes, IDs, masks, and padding policy."""
    raw, shard = load_shard_metadata(path)
    if shard.tokenizer_fingerprint != expected_fingerprint:
        raise ValueError("packed tokenizer fingerprint mismatch")
    if raw.get("packing_configuration_hash") != expected_config_hash:
        raise ValueError("packed configuration hash mismatch")
    if shard.stored_token_width != shard.context_length + 1:
        raise ValueError("stored token width must equal context length plus one")
    expected_token_bytes = shard.sample_count * shard.stored_token_width * TOKEN_DTYPE.itemsize
    expected_mask_bytes = shard.sample_count * shard.context_length * MASK_DTYPE.itemsize
    if shard.tokens_path.stat().st_size != expected_token_bytes:
        raise ValueError("packed token file size mismatch")
    if shard.loss_mask_path.stat().st_size != expected_mask_bytes:
        raise ValueError("packed loss-mask file size mismatch")
    checksums = raw.get("output_checksums", {})
    if not isinstance(checksums, dict):
        raise ValueError("packed output checksums are invalid")
    for binary in (shard.tokens_path, shard.loss_mask_path):
        if checksums.get(binary.name) != sha256_file(binary):
            raise ValueError(f"packed checksum failed: {binary.name}")
    tokens = np.memmap(
        shard.tokens_path, dtype=TOKEN_DTYPE, mode="r",
        shape=(shard.sample_count, shard.stored_token_width),
    )
    masks = np.memmap(
        shard.loss_mask_path, dtype=MASK_DTYPE, mode="r",
        shape=(shard.sample_count, shard.context_length),
    )
    if tokens.size and int(tokens.max()) >= int(raw["vocabulary_size"]):
        raise ValueError("packed token ID exceeds the vocabulary")
    if masks.size and not bool(np.isin(masks, (0, 1)).all()):
        raise ValueError("loss masks must contain only zero and one")
    padding_targets = tokens[:, 1:] == int(raw["pad_token_id"])
    if bool(np.any(masks[padding_targets] != 0)):
        raise ValueError("padding target has active loss")
    return shard


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Atomically write safe metadata containing no decoded text."""
    atomic_write_json(path, metadata)


def validate_packed_manifest(
    manifest_path: Path, expected_fingerprint: str, expected_config_hash: str
) -> dict[str, Any]:
    """Validate every listed shard and enforce family/split source isolation."""
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("packing manifest must be an object")
    if raw.get("tokenizer_fingerprint") != expected_fingerprint:
        raise ValueError("packing manifest tokenizer fingerprint mismatch")
    if raw.get("packing_configuration_hash") != expected_config_hash:
        raise ValueError("packing manifest configuration hash mismatch")
    root = manifest_path.parent.parent
    seen_inputs: dict[str, tuple[str, str]] = {}
    sample_count = binary_bytes = 0
    groups: set[str] = set()
    for relative in raw.get("shard_metadata", []):
        metadata_path = root / str(relative)
        metadata, shard = load_shard_metadata(metadata_path)
        validate_shard(metadata_path, expected_fingerprint, expected_config_hash)
        expected_parts = (shard.family, shard.split)
        if expected_parts[0] not in {"pretraining", "instruction"}:
            raise ValueError("unknown packed family")
        if expected_parts[1] not in {"train", "validation", "test"}:
            raise ValueError("unknown packed split")
        groups.add(f"{shard.family}_{shard.split}")
        checksums = metadata.get("input_shard_checksums", {})
        if not isinstance(checksums, dict):
            raise ValueError("input shard checksums must be an object")
        for source in checksums:
            previous = seen_inputs.setdefault(str(source), expected_parts)
            if previous != expected_parts:
                raise ValueError("input shard crosses packed family or split boundaries")
        sample_count += shard.sample_count
        binary_bytes += shard.tokens_path.stat().st_size + shard.loss_mask_path.stat().st_size
    return {
        "passed": True,
        "validated_shards": len(raw.get("shard_metadata", [])),
        "validated_samples": sample_count,
        "binary_bytes": binary_bytes,
        "groups": sorted(groups),
        "split_contamination_detected": False,
        "tokenizer_fingerprint": expected_fingerprint,
    }
