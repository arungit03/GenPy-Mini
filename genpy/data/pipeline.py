"""Incremental, resumable orchestration for the Step 2 text pipeline."""

import hashlib
import itertools
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional

from genpy.config import DataPipelineConfig

from .cleaning import assess_quality, normalize_text
from .dedup import ExactDeduplicator, content_hash
from .schema import GenPyDocument
from .split import assign_split
from .source import load_source_rows
from .stats import DatasetStats
from .writer import DocumentShardWriter


PIPELINE_VERSION = "genpy-step2-v1"


@dataclass
class PipelineResult:
    stats: DatasetStats
    output_dir: Path
    manifest_path: Path
    state_path: Path
    shard_files: List[str]
    completed: bool


def _fingerprint(config: DataPipelineConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_summary(config: DataPipelineConfig) -> dict:
    return asdict(config)


def _safe_metadata(row: Mapping[str, object], key: str, preserve: bool) -> Optional[str]:
    if not preserve:
        return None
    value = row.get(key)
    return str(value) if value is not None and isinstance(value, (str, int, float)) else None


def _document_from_row(
    row: Mapping[str, object], config: DataPipelineConfig, normalized: str, digest: str, split: str
) -> GenPyDocument:
    preserve = config.metadata.preserve_source_metadata
    stable_id = row.get("id", row.get("doc_id"))
    if stable_id is None or not isinstance(stable_id, (str, int)):
        stable_id = digest
    doc_id = str(stable_id)
    quality_value = row.get("quality_score") if preserve else None
    quality_score = float(quality_value) if isinstance(quality_value, (int, float)) else None
    return GenPyDocument(
        doc_id=doc_id,
        text=normalized,
        content_hash=digest,
        source_dataset=config.dataset.name,
        source_config=config.dataset.config,
        source_url=_safe_metadata(row, "url", preserve),
        source_dump=_safe_metadata(row, "dump", preserve),
        language=_safe_metadata(row, "language", preserve),
        quality_score=quality_score,
        char_count=len(normalized),
        byte_count=len(normalized.encode("utf-8")),
        split=split,
    )


def _load_state(state_path: Path) -> dict:
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise ValueError(f"Resume state must be a JSON object: {state_path}")
    return state


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        temporary.replace(path)
    except PermissionError:
        # Some Windows/OneDrive configurations briefly lock an existing target.
        # Preserve the atomic rename path when possible, with a safe writable
        # fallback so resumable state is not lost on developer machines.
        path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.unlink()


def _manifest(
    config: DataPipelineConfig,
    stats: DatasetStats,
    shard_files: List[str],
    completed: bool,
    created_at: str,
    skip_documents: int,
    max_documents: Optional[int],
) -> dict:
    source_end_index_exclusive = (
        skip_documents + max_documents if max_documents is not None else None
    )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "created_at": created_at,
        "python_version": platform.python_version(),
        "dataset": _config_summary(config),
        "skip_documents": skip_documents,
        "source_start_index": skip_documents,
        "requested_source_documents": max_documents,
        "source_end_index_exclusive": source_end_index_exclusive,
        "source_range": {
            "start_index": skip_documents,
            "requested_documents": max_documents,
            "end_index_exclusive": source_end_index_exclusive,
        },
        "statistics": stats.to_dict(),
        "generated_shard_filenames": sorted(shard_files),
        "completion_status": "complete" if completed else "incomplete",
    }


def _state(
    config: DataPipelineConfig,
    stats: DatasetStats,
    deduplicator: ExactDeduplicator,
    shard_files: List[str],
    completed: bool,
    fingerprint: str,
    manifest_filename: str,
    skip_documents: int,
    max_documents: Optional[int],
) -> dict:
    source_end_index_exclusive = (
        skip_documents + max_documents if max_documents is not None else None
    )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "configuration_fingerprint": fingerprint,
        "skip_documents": skip_documents,
        "source_start_index": skip_documents,
        "requested_source_documents": max_documents,
        "source_end_index_exclusive": source_end_index_exclusive,
        "source_documents_seen": stats.source_documents_seen,
        "statistics": stats.to_dict(),
        "seen_content_hashes": sorted(deduplicator.hashes),
        "completed_shard_files": sorted(shard_files),
        "completion_status": "complete" if completed else "incomplete",
        "manifest_filename": manifest_filename,
        "configuration": _config_summary(config),
    }


def run_pipeline(
    config: DataPipelineConfig,
    source: Optional[Iterable[Mapping[str, object]]] = None,
    max_documents: Optional[int] = None,
    output_dir: Optional[Path] = None,
    resume: bool = False,
    skip_documents: int = 0,
) -> PipelineResult:
    """Process rows incrementally, optionally resuming a prior compatible run."""
    if max_documents is not None and max_documents < 0:
        raise ValueError("max_documents must be non-negative")
    if skip_documents < 0:
        raise ValueError("skip_documents must be non-negative")
    processed_dir = Path(output_dir or config.output.processed_dir)
    manifest_dir = (
        processed_dir.parent / "manifests" if output_dir is not None else Path(config.output.manifest_dir)
    )
    state_path = manifest_dir / "prepare-state.json"
    fingerprint = _fingerprint(config)
    stats = DatasetStats()
    deduplicator = ExactDeduplicator()
    manifest_filename = f"prepare-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    prior_state = None

    if resume and state_path.exists():
        prior_state = _load_state(state_path)
        if prior_state.get("configuration_fingerprint") != fingerprint:
            raise ValueError("Resume configuration is incompatible with the saved pipeline state")
        saved_skip_documents = prior_state.get("skip_documents", 0)
        if saved_skip_documents != skip_documents:
            raise ValueError(
                "Resume source range is incompatible: skip_documents differs from the saved pipeline state"
            )
        stats = DatasetStats.from_dict(prior_state.get("statistics", {}))
        deduplicator = ExactDeduplicator(set(prior_state.get("seen_content_hashes", [])))
        manifest_filename = str(prior_state.get("manifest_filename", manifest_filename))
        if prior_state.get("completion_status") == "complete":
            manifest_path = manifest_dir / manifest_filename
            return PipelineResult(
                stats, processed_dir, manifest_path, state_path,
                list(prior_state.get("completed_shard_files", [])), True,
            )
    elif not resume and list(processed_dir.glob("*.jsonl.gz")):
        raise FileExistsError(
            f"Output directory already contains shards: {processed_dir}. Use --resume or a new output directory."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    writer = DocumentShardWriter(processed_dir, config.output.shard_max_documents)
    prior_seen = stats.source_documents_seen
    rows: Iterable[Mapping[str, object]] = source if source is not None else load_source_rows(config)
    rows = itertools.islice(rows, skip_documents, None)
    completed = False
    existing_shards = [path.name for path in processed_dir.glob("*.jsonl.gz")]
    created_at = prior_state.get("created_at", datetime.now(timezone.utc).isoformat()) if prior_state else datetime.now(timezone.utc).isoformat()

    def save_state() -> None:
        shard_files = sorted(set(existing_shards + writer.shard_files))
        _write_json(
            state_path,
            _state(
                config,
                stats,
                deduplicator,
                shard_files,
                completed,
                fingerprint,
                manifest_filename,
                skip_documents,
                max_documents,
            ),
        )

    try:
        for source_index, row in enumerate(rows):
            if source_index < prior_seen:
                continue
            if max_documents is not None and source_index >= max_documents:
                break
            stats.source_documents_seen += 1
            raw_text = row.get(config.dataset.text_field) if isinstance(row, Mapping) else None
            if not isinstance(raw_text, str):
                result = assess_quality(raw_text, config.processing.min_chars, config.processing.max_chars)
                stats.reject(result.reason or "invalid_text_type")
                save_state()
                continue
            normalized = normalize_text(raw_text, config.processing)
            result = assess_quality(normalized, config.processing.min_chars, config.processing.max_chars)
            if not result.accepted:
                stats.reject(result.reason or "rejected")
                save_state()
                continue
            digest = content_hash(normalized)
            if config.processing.exact_deduplication and not deduplicator.add(digest):
                stats.duplicate_documents += 1
                save_state()
                continue
            split = assign_split(digest, config.split.validation_fraction, config.split.seed)
            document = _document_from_row(row, config, normalized, digest, split)
            writer.write(document)
            stats.accept(split, document.char_count, document.byte_count)
            save_state()
        completed = True
    finally:
        writer.close()
        save_state()

    shard_files = sorted(set(existing_shards + writer.shard_files))
    manifest_path = manifest_dir / manifest_filename
    _write_json(
        manifest_path,
        _manifest(
            config,
            stats,
            shard_files,
            completed,
            created_at,
            skip_documents,
            max_documents,
        ),
    )
    save_state()
    return PipelineResult(stats, processed_dir, manifest_path, state_path, shard_files, completed)
