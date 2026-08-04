"""Deterministic streaming token packing with explicit target masks."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from genpy.data.schemas import InstructionRecord, PretrainingRecord
from genpy.data.sharding import iter_shard_records
from genpy.tokenizer.corpus import verify_shard
from genpy.tokenizer.fingerprint import atomic_write_json, canonical_sha256, sha256_file
from genpy.tokenizer.serialization import serialize_instruction, serialize_pretraining
from genpy.tokenizer.tokenizer import GenPyTokenizer
from genpy.training.packed_format import MASK_DTYPE, TOKEN_DTYPE, write_binary, write_metadata

Family = Literal["pretraining", "instruction"]
Split = Literal["train", "validation", "test"]


class PackingError(RuntimeError):
    """Raised when source, tokenizer, storage, or packing contracts fail."""


@dataclass(frozen=True, slots=True)
class PackingConfig:
    """Validated paths and policies for one packing mode."""

    path: Path
    project_root: Path
    raw: dict[str, Any]
    config_hash: str

    @property
    def packing(self) -> dict[str, Any]:
        return dict(self.raw["packing"])

    @property
    def tokenizer(self) -> dict[str, Any]:
        return dict(self.raw["tokenizer"])

    @property
    def source(self) -> dict[str, Any]:
        return dict(self.raw["source"])

    @property
    def output_root(self) -> Path:
        return self.project_root / str(self.packing["output_root"])

    @property
    def tokenizer_artifact(self) -> Path:
        return self.project_root / str(self.tokenizer["artifact_path"])


@dataclass(frozen=True, slots=True)
class TokenizedRecord:
    """One canonical record represented only by IDs and target activity."""

    token_ids: tuple[int, ...]
    target_active: tuple[bool, ...]
    structural_tokens: int
    input_shard: str
    input_checksum: str


PACKING_KEYS = {
    "mode", "source_mode", "context_length", "stored_token_width", "token_dtype",
    "loss_mask_dtype", "seed", "retain_final_partial", "mask_cross_record_transitions",
    "maximum_shard_bytes", "maximum_output_bytes", "temporary_space_multiplier",
    "output_root", "report_json", "report_markdown",
}
TOKENIZER_KEYS = {"name", "version", "vocab_size", "artifact_path", "fingerprint"}
SOURCE_KEYS = {"dataset_version", "phase2_root", "fixture_path"}


def load_packing_config(path: Path, project_root: Path | None = None) -> PackingConfig:
    """Load strict packing settings and reject smoke/production crossover."""
    resolved = path.resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version", "packing", "tokenizer", "source", "loss_policies", "runtime"
    }:
        raise PackingError("packing config has unknown or missing top-level keys")
    for name in ("packing", "tokenizer", "source", "loss_policies", "runtime"):
        if not isinstance(raw[name], dict):
            raise PackingError(f"packing section must be a mapping: {name}")
    for value, allowed, name in (
        (raw["packing"], PACKING_KEYS, "packing"),
        (raw["tokenizer"], TOKENIZER_KEYS, "tokenizer"),
        (raw["source"], SOURCE_KEYS, "source"),
    ):
        if set(value) != allowed:
            raise PackingError(f"unknown or missing {name} keys")
    settings = raw["packing"]
    mode = settings["mode"]
    if mode not in {"smoke", "production"}:
        raise PackingError("packing mode must be smoke or production")
    expected_context = int(settings["context_length"]) if mode == "smoke" else 1024
    expected_vocab = int(raw["tokenizer"]["vocab_size"]) if mode == "smoke" else 16384
    if (mode == "smoke" and not 1 <= expected_context <= 64) or (
        mode == "production" and int(settings["context_length"]) != expected_context
    ):
        raise PackingError("packing context length violates its mode contract")
    if int(settings["stored_token_width"]) != expected_context + 1:
        raise PackingError("stored token width must be context length plus one")
    if int(raw["tokenizer"]["vocab_size"]) != expected_vocab:
        raise PackingError("packing tokenizer vocabulary violates its mode contract")
    if settings["token_dtype"] != "uint16_le" or settings["loss_mask_dtype"] != "uint8":
        raise PackingError("packed dtypes must be uint16_le and uint8")
    if mode == "smoke" and settings["source_mode"] != "safe_fixture":
        raise PackingError("smoke packing may use only safe fixtures")
    if mode == "production" and settings["source_mode"] != "phase2":
        raise PackingError("production packing may use only Phase 2 shards")
    if raw["loss_policies"] != {
        "pretraining": "full_lm", "instruction": "assistant_only"
    }:
        raise PackingError("default loss policies must be explicit")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    root = project_root or next(
        (parent for parent in resolved.parents if (parent / "pyproject.toml").is_file()),
        Path.cwd(),
    )
    return PackingConfig(
        path=resolved,
        project_root=root.resolve(),
        raw=raw,
        config_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _instruction_parts(record: InstructionRecord) -> tuple[str, str]:
    users = [message.content for message in record.messages if message.role == "user"]
    assistants = [message.content for message in record.messages if message.role == "assistant"]
    if len(users) != 1 or len(assistants) != 1:
        raise PackingError("instruction record must contain one user and one assistant")
    return users[0], assistants[0]


def _iter_fixture(config: PackingConfig, family: Family, split: Split) -> Iterator[dict[str, Any]]:
    fixture = config.project_root / str(config.source["fixture_path"])
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise PackingError("safe fixture must contain a list")
    for record in raw:
        if not isinstance(record, dict):
            raise PackingError("safe fixture records must be objects")
        if record.get("family") == family and record.get("split") == split:
            yield {
                **record,
                "_input_shard": f"{fixture.name}#{family}/{split}",
                "_input_checksum": sha256_file(fixture),
                "_expected_split": split,
            }


def _iter_phase2(config: PackingConfig, family: Family, split: Split) -> Iterator[dict[str, Any]]:
    directory = config.project_root / str(config.source["phase2_root"]) / family / split
    for shard in sorted(directory.glob("part-*.jsonl.zst"), key=lambda path: path.name):
        manifest = verify_shard(shard)
        for record in iter_shard_records((shard,)):
            yield {
                **record,
                "_input_shard": f"{family}/{split}/{shard.name}",
                "_input_checksum": str(manifest["sha256"]),
                "_expected_split": split,
            }


def iter_source_records(
    config: PackingConfig, family: Family, split: Split
) -> Iterator[dict[str, Any]]:
    """Stream one isolated family and split in stable source order."""
    if config.packing["source_mode"] == "safe_fixture":
        yield from _iter_fixture(config, family, split)
    else:
        yield from _iter_phase2(config, family, split)


def tokenize_record(
    tokenizer: GenPyTokenizer,
    family: Family,
    record: dict[str, Any],
    loss_policy: str,
) -> TokenizedRecord:
    """Reuse Phase 3 serialization and create target activity per serialized token."""
    if family == "pretraining":
        if "code" in record:
            code = str(record["code"])
        else:
            parsed = PretrainingRecord.from_dict(
                {key: value for key, value in record.items() if not key.startswith("_")}
            )
            if parsed.split != record.get("_expected_split"):
                raise PackingError("pretraining split mismatch")
            if not parsed.quality.secret_scan_passed or not parsed.quality.pii_scan_passed:
                raise PackingError("pretraining record failed Phase 2 safety flags")
            code = parsed.text
        serialized = serialize_pretraining(code)
        ids = tokenizer.encode_pretraining_record(code)
        active = [True] * len(ids)
    else:
        if "prompt" in record and "code" in record:
            prompt, code = str(record["prompt"]), str(record["code"])
        else:
            parsed_instruction = InstructionRecord.from_dict(
                {key: value for key, value in record.items() if not key.startswith("_")}
            )
            if parsed_instruction.split != record.get("_expected_split"):
                raise PackingError("instruction split mismatch")
            prompt, code = _instruction_parts(parsed_instruction)
        serialized = serialize_instruction(prompt, code)
        ids = tokenizer.encode_instruction_record(prompt, code)
        if loss_policy == "assistant_only":
            assistant_index = ids.index(tokenizer.special_token_ids["assistant"])
            active = [index >= assistant_index for index in range(len(ids))]
        elif loss_policy == "full_lm":
            active = [True] * len(ids)
        else:
            raise PackingError(f"unsupported loss policy: {loss_policy}")
    if not ids or any(token_id < 0 or token_id >= tokenizer.vocab_size for token_id in ids):
        raise PackingError("record produced invalid token IDs")
    active[0] = False
    return TokenizedRecord(
        token_ids=tuple(ids),
        target_active=tuple(active),
        structural_tokens=serialized.structural_tokens,
        input_shard=str(record["_input_shard"]),
        input_checksum=str(record["_input_checksum"]),
    )


def _pack_records(
    records: Iterator[TokenizedRecord], context: int, pad_id: int, retain_final: bool
) -> tuple[list[np.ndarray], list[np.ndarray], Counter[str], dict[str, str]]:
    token_buffer: list[int] = []
    active_buffer: list[bool] = []
    token_samples: list[np.ndarray] = []
    mask_samples: list[np.ndarray] = []
    stats: Counter[str] = Counter()
    input_checksums: dict[str, str] = {}
    width = context + 1
    for record in records:
        if token_buffer:
            stats["record_boundaries"] += 1
            stats["cross_record_transitions_masked"] += 1
        token_buffer.extend(record.token_ids)
        active_buffer.extend(record.target_active)
        stats["input_record_count"] += 1
        stats["real_tokens"] += len(record.token_ids)
        stats["structural_tokens"] += record.structural_tokens
        if len(record.token_ids) > context:
            stats["records_longer_than_context"] += 1
        input_checksums[record.input_shard] = record.input_checksum
        while len(token_buffer) >= width:
            token_samples.append(np.asarray(token_buffer[:width], dtype=TOKEN_DTYPE))
            mask_samples.append(np.asarray(active_buffer[1:width], dtype=MASK_DTYPE))
            stats["active_loss_targets"] += sum(active_buffer[1:width])
            stats["masked_targets"] += context - sum(active_buffer[1:width])
            token_buffer = token_buffer[context:]
            active_buffer = active_buffer[context:]
    if retain_final and len(token_buffer) > 1:
        real_width = len(token_buffer)
        padding = width - real_width
        token_samples.append(
            np.asarray(token_buffer + [pad_id] * padding, dtype=TOKEN_DTYPE)
        )
        final_mask = active_buffer[1:] + [False] * (context - (real_width - 1))
        mask_samples.append(np.asarray(final_mask, dtype=MASK_DTYPE))
        stats["padding_tokens"] += padding
        stats["active_loss_targets"] += sum(final_mask)
        stats["masked_targets"] += context - sum(final_mask)
    stats["packed_sample_count"] = len(token_samples)
    return token_samples, mask_samples, stats, input_checksums


def _write_group(
    config: PackingConfig,
    tokenizer: GenPyTokenizer,
    family: Family,
    split: Split,
    tokens: list[np.ndarray],
    masks: list[np.ndarray],
    stats: Counter[str],
    input_checksums: dict[str, str],
) -> list[str]:
    context = int(config.packing["context_length"])
    sample_bytes = (context + 1) * TOKEN_DTYPE.itemsize + context * MASK_DTYPE.itemsize
    per_shard = max(1, int(config.packing["maximum_shard_bytes"]) // sample_bytes)
    output = config.output_root / family / split
    output.mkdir(parents=True, exist_ok=True)
    metadata_paths: list[str] = []
    for shard_index, start in enumerate(range(0, len(tokens), per_shard)):
        token_array = np.stack(tokens[start : start + per_shard])
        mask_array = np.stack(masks[start : start + per_shard])
        stem = f"{split}-{family}-{shard_index:05d}"
        token_path = output / f"{stem}.tokens.bin"
        mask_path = output / f"{stem}.lossmask.bin"
        meta_path = output / f"{stem}.meta.json"
        write_binary(token_path, token_array)
        write_binary(mask_path, mask_array)
        metadata = {
            "family": family,
            "split": split,
            "shard_number": shard_index,
            "sample_count": int(token_array.shape[0]),
            "real_token_count": int(np.count_nonzero(token_array != 0)),
            "structural_token_count": int(np.isin(token_array, (1, 2, 3, 4, 5, 6)).sum()),
            "padding_token_count": int(np.count_nonzero(token_array == 0)),
            "active_loss_target_count": int(mask_array.sum()),
            "input_sequence_length": context,
            "stored_token_width": context + 1,
            "token_dtype": "little-endian uint16",
            "loss_mask_dtype": "uint8",
            "vocabulary_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.special_token_ids["pad"],
            "tokenizer_name": config.tokenizer["name"],
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "dataset_version": config.source["dataset_version"],
            "input_shard_checksums": input_checksums,
            "packing_configuration_hash": config.config_hash,
            "creation_timestamp_utc": datetime.now(UTC).isoformat(),
            "tokens_file": token_path.name,
            "loss_mask_file": mask_path.name,
            "output_checksums": {
                token_path.name: sha256_file(token_path),
                mask_path.name: sha256_file(mask_path),
            },
            "loss_policy": config.raw["loss_policies"][family],
            "resume_state_identifier": canonical_sha256(
                {"config": config.config_hash, "family": family, "split": split}
            ),
        }
        write_metadata(meta_path, metadata)
        metadata_paths.append(str(meta_path.relative_to(config.output_root)).replace("\\", "/"))
    return metadata_paths


def prepare_packed_data(
    config: PackingConfig,
    *,
    dry_run: bool = False,
    force: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Pack all six isolated family/split groups or estimate them without writes."""
    if config.packing["mode"] == "production" and (
        config.tokenizer["fingerprint"] == "populated_after_training"
        or not config.tokenizer_artifact.is_dir()
    ):
        raise PackingError("production packing requires the frozen production tokenizer")
    tokenizer = GenPyTokenizer.load(config.tokenizer_artifact)
    if tokenizer.fingerprint != config.tokenizer["fingerprint"]:
        raise PackingError("packing tokenizer fingerprint mismatch")
    if tokenizer.vocab_size != int(config.tokenizer["vocab_size"]):
        raise PackingError("packing tokenizer vocabulary mismatch")
    previous_binary_checksums: dict[str, str] = {}
    resume_events = 0
    if config.output_root.exists() and not dry_run:
        generated = [
            path
            for path in config.output_root.rglob("*")
            if path.is_file() and path.name != ".gitkeep"
        ]
        if generated and not force:
            manifest_path = config.output_root / "manifests/packing_manifest.json"
            if manifest_path.is_file():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    isinstance(existing, dict)
                    and existing.get("packing_configuration_hash") == config.config_hash
                ):
                    return {
                        **existing,
                        "resume_events": int(existing.get("resume_events", 0)) + 1,
                    }
            if not resume:
                raise FileExistsError("incomplete packed output exists and resume is disabled")
            quarantine = config.output_root.with_name(
                config.output_root.name + f".incomplete-{config.config_hash[:12]}"
            )
            if quarantine.exists():
                raise FileExistsError("incomplete packed-output quarantine already exists")
            config.output_root.replace(quarantine)
            resume_events = 1
        if force:
            if config.packing["mode"] != "smoke":
                raise PackingError("production packed output cannot be replaced in place")
            previous_binary_checksums = {
                str(path.relative_to(config.output_root)).replace("\\", "/"): sha256_file(path)
                for path in sorted(config.output_root.rglob("*.bin"))
            }
            shutil.rmtree(config.output_root)
    tracemalloc.start()
    started = time.perf_counter()
    groups: dict[str, Any] = {}
    all_metadata: list[str] = []
    estimated_bytes = 0
    for family in ("pretraining", "instruction"):
        for split in ("train", "validation", "test"):
            typed_family: Family = family
            typed_split: Split = split
            records = (
                tokenize_record(
                    tokenizer,
                    typed_family,
                    record,
                    str(config.raw["loss_policies"][family]),
                )
                for record in iter_source_records(config, typed_family, typed_split)
            )
            tokens, masks, stats, checksums = _pack_records(
                records,
                int(config.packing["context_length"]),
                tokenizer.special_token_ids["pad"],
                bool(config.packing["retain_final_partial"]),
            )
            bytes_count = sum(array.nbytes for array in tokens) + sum(
                array.nbytes for array in masks
            )
            estimated_bytes += bytes_count
            key = f"{family}_{split}"
            groups[key] = {
                **dict(stats),
                "output_binary_bytes": bytes_count,
                "packing_efficiency": (
                    stats["real_tokens"]
                    / (stats["packed_sample_count"] * (int(config.packing["context_length"]) + 1))
                    if stats["packed_sample_count"]
                    else 0.0
                ),
                "padding_percentage": (
                    100.0 * stats["padding_tokens"]
                    / (stats["packed_sample_count"] * (int(config.packing["context_length"]) + 1))
                    if stats["packed_sample_count"]
                    else 0.0
                ),
            }
            if not dry_run and tokens:
                all_metadata.extend(
                    _write_group(
                        config, tokenizer, typed_family, typed_split,
                        tokens, masks, stats, checksums,
                    )
                )
    if estimated_bytes > int(config.packing["maximum_output_bytes"]):
        raise PackingError("estimated packed output exceeds the configured safety limit")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    current_binary_checksums = (
        {
            str(path.relative_to(config.output_root)).replace("\\", "/"): sha256_file(path)
            for path in sorted(config.output_root.rglob("*.bin"))
        }
        if not dry_run
        else {}
    )
    determinism = (
        "passed"
        if previous_binary_checksums and previous_binary_checksums == current_binary_checksums
        else "pending_repeat_check" if not dry_run else "dry_run"
    )
    if previous_binary_checksums and determinism != "passed":
        raise PackingError("repeated smoke packing produced different binary output")
    report = {
        "mode": config.packing["mode"],
        "source_dataset_version": config.source["dataset_version"],
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "packing_configuration_hash": config.config_hash,
        "context_length": config.packing["context_length"],
        "stored_token_width": config.packing["stored_token_width"],
        "groups": groups,
        "shard_metadata": all_metadata,
        "estimated_or_actual_binary_bytes": estimated_bytes,
        "processing_seconds": round(time.perf_counter() - started, 6),
        "peak_memory_bytes": peak,
        "resume_events": resume_events,
        "checksum_failures": 0,
        "rejected_records": 0,
        "determinism_check": determinism,
        "limitations": ["Smoke output uses original safe fixtures only."]
        if config.packing["mode"] == "smoke" else [],
    }
    if not dry_run:
        manifest = config.output_root / "manifests/packing_manifest.json"
        atomic_write_json(manifest, report)
        report_json = config.project_root / str(config.packing["report_json"])
        atomic_write_json(report_json, report)
    return report


def estimate_production_storage(config: PackingConfig, token_count: int) -> dict[str, Any]:
    """Estimate binary and temporary bytes from an exact serialized token count."""
    context = int(config.packing["context_length"])
    samples = math.ceil(max(0, token_count - 1) / context) if token_count else 0
    token_bytes = samples * (context + 1) * 2
    mask_bytes = samples * context
    output = token_bytes + mask_bytes
    temporary = math.ceil(output * float(config.packing["temporary_space_multiplier"]))
    free = shutil.disk_usage(config.output_root.parent).free
    return {
        "exact_input_tokens": token_count,
        "estimated_samples": samples,
        "estimated_token_bytes": token_bytes,
        "estimated_loss_mask_bytes": mask_bytes,
        "estimated_output_bytes": output,
        "estimated_temporary_space_bytes": temporary,
        "available_disk_bytes": free,
        "within_configured_limit": output <= int(config.packing["maximum_output_bytes"]),
        "within_available_disk": output + temporary <= free,
    }
