"""Deterministic streaming JSONL with Zstandard sharding."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import zstandard


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_shards(
    records: Iterable[dict[str, Any]],
    output_dir: Path,
    *,
    maximum_uncompressed_bytes: int,
    compression_level: int,
    config_hash: str,
) -> list[dict[str, Any]]:
    """Write stable atomically-renamed shards and return shard manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.glob("part-*.jsonl.zst*"):
        if old_path.is_file():
            old_path.unlink()
    manifests: list[dict[str, Any]] = []
    shard_index = 0
    compressor = zstandard.ZstdCompressor(level=compression_level)
    handle: Any = None
    stream: Any = None
    temporary: Path | None = None
    final: Path | None = None
    record_count = 0
    uncompressed_size = 0
    sources: Counter[str] = Counter()
    licences: Counter[str] = Counter()

    def close_shard() -> None:
        nonlocal handle, stream, temporary, final
        nonlocal record_count, uncompressed_size, sources, licences
        if stream is None or handle is None or temporary is None or final is None:
            return
        stream.flush(zstandard.FLUSH_FRAME)
        stream.close()
        handle.close()
        temporary.replace(final)
        manifest = {
            "shard": final.name,
            "record_count": record_count,
            "compressed_bytes": final.stat().st_size,
            "uncompressed_bytes": uncompressed_size,
            "sha256": _sha256(final),
            "source_composition": dict(sorted(sources.items())),
            "licence_composition": dict(sorted(licences.items())),
            "created_at": datetime.now(UTC).isoformat(),
            "pipeline_configuration_sha256": config_hash,
        }
        manifest_path = final.with_suffix(final.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifests.append(manifest)
        handle = stream = temporary = final = None
        record_count = uncompressed_size = 0
        sources = Counter()
        licences = Counter()

    for record in records:
        line = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        if (
            stream is not None
            and record_count
            and uncompressed_size + len(line) > maximum_uncompressed_bytes
        ):
            close_shard()
            shard_index += 1
        if stream is None:
            final = output_dir / f"part-{shard_index:05d}.jsonl.zst"
            temporary = final.with_suffix(final.suffix + ".tmp")
            handle = temporary.open("wb")
            stream = compressor.stream_writer(handle, closefd=False)
        stream.write(line)
        record_count += 1
        uncompressed_size += len(line)
        sources[str(record.get("source_id", "unknown"))] += 1
        licences[str(record.get("licence_spdx", "unknown"))] += 1
    close_shard()
    return manifests


def iter_shard_records(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Yield decoded records from Zstandard JSONL shards."""
    decompressor = zstandard.ZstdDecompressor()
    for path in paths:
        with path.open("rb") as compressed:
            with decompressor.stream_reader(compressed) as reader:
                pending = b""
                while chunk := reader.read(1024 * 1024):
                    pending += chunk
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    for line in lines:
                        if line:
                            value = json.loads(line)
                            if not isinstance(value, dict):
                                raise ValueError(f"non-object record in {path}")
                            yield value
                if pending.strip():
                    value = json.loads(pending)
                    if not isinstance(value, dict):
                        raise ValueError(f"non-object record in {path}")
                    yield value
