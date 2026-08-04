"""Streaming tokenizer-corpus selection from Phase 2 training shards."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from genpy.data.schemas import InstructionRecord, PretrainingRecord
from genpy.data.sharding import iter_shard_records
from genpy.tokenizer.config import TokenizerConfig
from genpy.tokenizer.fingerprint import atomic_write_json, canonical_sha256, sha256_file
from genpy.tokenizer.serialization import serialize_instruction, serialize_pretraining

Family = Literal["pretraining", "instruction"]


class CorpusError(ValueError):
    """Raised when Phase 2 data violates tokenizer-training rules."""


@dataclass(frozen=True, slots=True)
class CorpusItem:
    """One selected serialized training sample and safe manifest metadata."""

    record_id: str
    content_sha256: str
    source_shard: str
    family: Family
    split: str
    serialized_bytes: int
    source_id: str
    licence_spdx: str
    text: str

    def manifest_dict(self) -> dict[str, Any]:
        """Return metadata without raw corpus text."""
        value = asdict(self)
        del value["text"]
        return value


def verify_shard(path: Path) -> dict[str, Any]:
    """Verify one Phase 2 shard against its sidecar checksum."""
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise CorpusError(f"missing shard manifest: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("sha256") != sha256_file(path):
        raise CorpusError(f"shard checksum mismatch: {path.name}")
    return manifest


def _instruction_parts(record: InstructionRecord) -> tuple[str, str, str | None]:
    users = [message.content for message in record.messages if message.role == "user"]
    assistants = [message.content for message in record.messages if message.role == "assistant"]
    systems = [message.content for message in record.messages if message.role == "system"]
    if len(users) != 1 or len(assistants) != 1 or len(systems) > 1:
        raise CorpusError("instruction record must contain one user and one assistant message")
    return users[0], assistants[0], systems[0] if systems else None


def iter_family_items(config: TokenizerConfig, family: Family) -> Iterator[CorpusItem]:
    """Yield canonical training-split samples in stable shard/record order."""
    directory_key = f"{family}_train"
    directory = config.resolve(str(config.corpus[directory_key]))
    for shard in sorted(directory.glob("part-*.jsonl.zst"), key=lambda item: item.name):
        verify_shard(shard)
        for value in iter_shard_records((shard,)):
            if family == "pretraining":
                record = PretrainingRecord.from_dict(value)
                if record.split != "train":
                    raise CorpusError(
                        f"non-training record found in tokenizer shard: {record.record_id}"
                    )
                if not record.quality.secret_scan_passed or not record.quality.pii_scan_passed:
                    raise CorpusError(
                        f"unapproved safety flags in tokenizer shard: {record.record_id}"
                    )
                serialized = serialize_pretraining(record.text)
                source_id = record.source_id
                licence = record.licence_spdx
                record_id = record.record_id
                digest = record.content_sha256
                split = record.split
            else:
                instruction = InstructionRecord.from_dict(value)
                if instruction.split != "train":
                    raise CorpusError(
                        f"non-training record found in tokenizer shard: {instruction.record_id}"
                    )
                prompt, code, system = _instruction_parts(instruction)
                serialized = serialize_instruction(prompt, code, system)
                source_id = instruction.source_id
                licence = instruction.licence_spdx
                record_id = instruction.record_id
                digest = instruction.content_sha256
                split = instruction.split
            yield CorpusItem(
                record_id=record_id,
                content_sha256=digest,
                source_shard=shard.name,
                family=family,
                split=split,
                serialized_bytes=len(serialized.text.encode("utf-8")),
                source_id=source_id,
                licence_spdx=licence,
                text=serialized.text,
            )


def iter_selected_items(
    config: TokenizerConfig,
    *,
    maximum_bytes: int | None = None,
    maximum_records: int | None = None,
    source_ids: set[str] | None = None,
) -> Iterator[CorpusItem]:
    """Apply deterministic family byte budgets without duplicating records."""
    total_budget = int(maximum_bytes or config.corpus["maximum_bytes"])
    mixture = config.corpus["mixture"]
    budgets = {
        "pretraining": int(total_budget * float(mixture["pretraining_weight"])),
        "instruction": int(total_budget * float(mixture["instruction_weight"])),
    }
    selected_count = 0
    for family in ("pretraining", "instruction"):
        used = 0
        for item in iter_family_items(config, family):
            if source_ids is not None and item.source_id not in source_ids:
                continue
            if maximum_records is not None and selected_count >= maximum_records:
                return
            if used + item.serialized_bytes > budgets[family]:
                continue
            yield item
            used += item.serialized_bytes
            selected_count += 1


def prepare_corpus_manifest(
    config: TokenizerConfig,
    *,
    maximum_bytes: int | None = None,
    maximum_records: int | None = None,
    source_ids: set[str] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Write a deterministic metadata-only corpus manifest and summary."""
    manifest_path = output or config.resolve(str(config.corpus["manifest_path"]))
    summary_path = config.resolve(str(config.corpus["summary_path"]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    digest_lines: list[str] = []
    family_bytes: Counter[str] = Counter()
    family_records: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    licences: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in iter_selected_items(
            config,
            maximum_bytes=maximum_bytes,
            maximum_records=maximum_records,
            source_ids=source_ids,
        ):
            line = json.dumps(item.manifest_dict(), sort_keys=True, separators=(",", ":"))
            handle.write(line + "\n")
            digest_lines.append(line)
            family_bytes[item.family] += item.serialized_bytes
            family_records[item.family] += 1
            sources[item.source_id] += 1
            licences[item.licence_spdx] += 1
    temporary.replace(manifest_path)
    total_bytes = sum(family_bytes.values())
    corpus_fingerprint = canonical_sha256(
        {"manifest_lines": digest_lines, "configuration_sha256": config.config_hash}
    )
    summary: dict[str, Any] = {
        "corpus_fingerprint": corpus_fingerprint,
        "configuration_sha256": config.config_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "selected_records": sum(family_records.values()),
        "training_bytes": total_bytes,
        "family_records": dict(sorted(family_records.items())),
        "family_bytes": dict(sorted(family_bytes.items())),
        "actual_mixture": {
            family: bytes_count / total_bytes if total_bytes else 0.0
            for family, bytes_count in sorted(family_bytes.items())
        },
        "source_composition": dict(sorted(sources.items())),
        "licence_composition": dict(sorted(licences.items())),
        "selection_seed": int(config.tokenizer["training_seed"]),
        "selection_order": "family_then_shard_name_then_record_order",
        "contains_raw_text": False,
    }
    atomic_write_json(summary_path, summary)
    return summary


def iter_manifest_training_text(
    config: TokenizerConfig, manifest_path: Path | None = None
) -> Iterator[str]:
    """Replay only manifest-selected records and verify identity, hash, split, and size."""
    path = manifest_path or config.resolve(str(config.corpus["manifest_path"]))
    selected: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("split") != "train":
            raise CorpusError("tokenizer manifest contains non-training data")
        selected[str(value["record_id"])] = value
    seen: set[str] = set()
    for family in ("pretraining", "instruction"):
        for item in iter_family_items(config, family):
            expected = selected.get(item.record_id)
            if expected is None:
                continue
            if (
                expected.get("content_sha256") != item.content_sha256
                or expected.get("family") != item.family
                or int(expected.get("serialized_bytes", -1)) != item.serialized_bytes
            ):
                raise CorpusError(f"manifest identity mismatch for record {item.record_id}")
            seen.add(item.record_id)
            yield item.text
    missing = set(selected) - seen
    if missing:
        raise CorpusError(f"manifest references {len(missing)} missing records")
