"""End-to-end bounded-memory GenPy Phase 2 data pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from genpy.data.downloader import DownloadResult, iter_source_files, prepare_source
from genpy.data.exact_dedup import ExactDeduplicator
from genpy.data.licences import LicencePolicy
from genpy.data.near_dedup import NearDeduplicator
from genpy.data.normalize import NormalizationError, normalize_python_bytes
from genpy.data.pii_scan import scan_pii
from genpy.data.python_validation import validate_python
from genpy.data.quality import has_generated_marker, score_quality
from genpy.data.safety_filter import scan_unsafe_content
from genpy.data.schemas import PretrainingRecord, QualityInfo, content_sha256, stable_record_id
from genpy.data.secret_scan import scan_secrets
from genpy.data.sharding import write_shards
from genpy.data.source_registry import SourceEntry, SourceRegistry
from genpy.data.splitting import leakage_report, split_pretraining_records
from genpy.data.statistics import write_dataset_report

LOGGER = logging.getLogger(__name__)

SAFE_SCANNER_CONTROLS = (
    "value = 42\nprint(value)\n",
    "def add(left, right):\n    return left + right\n",
    "items = [1, 2, 3]\nprint(sum(items))\n",
    "name = input().strip()\nprint(f'Hello {name}')\n",
    "for index in range(5):\n    print(index)\n",
)


def scanner_false_positive_control() -> dict[str, float | int]:
    """Measure scanner flags on a small, explicitly fictional safe control set."""
    flagged = sum(
        bool(scan_secrets(text) or scan_pii(text) or scan_unsafe_content(text))
        for text in SAFE_SCANNER_CONTROLS
    )
    return {
        "safe_control_records": len(SAFE_SCANNER_CONTROLS),
        "false_positive_records": flagged,
        "observed_false_positive_rate": flagged / len(SAFE_SCANNER_CONTROLS),
    }


def load_phase2_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate Phase 2 YAML configuration."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Phase 2 config must be a mapping")
    for key in (
        "paths",
        "ingestion",
        "normalization",
        "quality",
        "deduplication",
        "splits",
        "sharding",
    ):
        if key not in value or not isinstance(value[key], dict):
            raise ValueError(f"Phase 2 config requires mapping: {key}")
    return value


def _resolve(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def _configuration_hash(config: dict[str, Any], overrides: dict[str, Any]) -> str:
    payload = json.dumps({"config": config, "overrides": overrides}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _safe_reset_database(path: Path) -> None:
    for candidate in (path, path.with_name(path.name + "-shm"), path.with_name(path.name + "-wal")):
        if candidate.is_file():
            candidate.unlink()


class DatasetPipeline:
    """Build cleaned, deduplicated, split, and sharded pretraining data."""

    def __init__(
        self,
        config_path: Path,
        *,
        project_root: Path | None = None,
        source_ids: set[str] | None = None,
        maximum_records: int | None = None,
        maximum_download_bytes: int | None = None,
        seed: int | None = None,
        output_directory: Path | None = None,
        workers: int | None = None,
        resume: bool | None = None,
    ) -> None:
        self.config_path = config_path.resolve()
        self.project_root = (project_root or Path.cwd()).resolve()
        self.config = load_phase2_config(self.config_path)
        self.source_ids = source_ids
        self.maximum_records = maximum_records
        self.maximum_download_bytes = maximum_download_bytes
        self.seed = int(seed if seed is not None else self.config["seed"])
        self.workers = int(workers if workers is not None else self.config["ingestion"]["workers"])
        self.resume = bool(
            resume if resume is not None else self.config["ingestion"].get("resume", True)
        )
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        self.paths = {
            name: _resolve(str(value), self.project_root)
            for name, value in self.config["paths"].items()
        }
        if output_directory is not None:
            output_root = output_directory.resolve()
            self.paths.update(
                {
                    "cleaned": output_root / "cleaned",
                    "splits": output_root / "splits",
                    "manifests": output_root / "manifests",
                    "quarantine": output_root / "quarantine",
                    "reports": output_root / "reports",
                    "work": output_root / ".work",
                }
            )
        overrides = {
            "source_ids": sorted(source_ids) if source_ids else None,
            "maximum_records": maximum_records,
            "maximum_download_bytes": maximum_download_bytes,
            "seed": self.seed,
            "output_directory": str(output_directory) if output_directory else None,
            "workers": self.workers,
        }
        self.config_hash = _configuration_hash(self.config, overrides)

    def _quarantine(
        self,
        *,
        reason: str,
        source: SourceEntry,
        path: Path,
        digest: str,
        categories: tuple[str, ...] = (),
    ) -> None:
        self.paths["quarantine"].mkdir(parents=True, exist_ok=True)
        manifest = self.paths["quarantine"] / "quarantine_manifest.jsonl"
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "reason": reason,
                        "categories": list(categories),
                        "record_hash": digest,
                        "source_id": source.id,
                        "file_path": path.as_posix(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def _process_file(
        self,
        source: SourceEntry,
        root: Path,
        path: Path,
        policy: LicencePolicy,
        counters: Counter[str],
        rejections: Counter[str],
    ) -> PretrainingRecord | None:
        counters["downloaded"] += 1
        allowed, reason = policy.decision(
            source.dataset_level_licence,
            provenance_available=source.provenance_available,
            dataset_licence=source.dataset_level_licence,
        )
        if not allowed:
            rejections[reason or "unknown_licence"] += 1
            return None
        counters["licence_accepted"] += 1
        maximum_bytes = int(self.config["normalization"]["maximum_bytes"])
        with path.open("rb") as handle:
            raw = handle.read(maximum_bytes + 1)
        raw_digest = hashlib.sha256(raw).hexdigest()
        try:
            text = normalize_python_bytes(
                raw,
                minimum_bytes=int(self.config["normalization"]["minimum_bytes"]),
                maximum_bytes=maximum_bytes,
                final_newline=bool(self.config["normalization"]["final_newline"]),
            )
        except NormalizationError as error:
            rejections[error.reason] += 1
            self._quarantine(reason=error.reason, source=source, path=path, digest=raw_digest)
            return None
        counters["utf8_valid"] += 1
        secret_categories = scan_secrets(text)
        pii_categories = scan_pii(text)
        if secret_categories:
            rejections["secret_detected"] += 1
            self._quarantine(
                reason="secret_detected",
                source=source,
                path=path,
                digest=raw_digest,
                categories=secret_categories,
            )
            return None
        if pii_categories:
            rejections["pii_detected"] += 1
            self._quarantine(
                reason="pii_detected",
                source=source,
                path=path,
                digest=raw_digest,
                categories=pii_categories,
            )
            return None
        unsafe_categories = scan_unsafe_content(text)
        if unsafe_categories:
            rejections["unsafe_content"] += 1
            self._quarantine(
                reason="unsafe_content",
                source=source,
                path=path,
                digest=raw_digest,
                categories=unsafe_categories,
            )
            return None
        counters["secret_pii_safe"] += 1
        validation = validate_python(text)
        if not validation.ast_valid or not validation.tokenize_valid:
            rejections["invalid_python"] += 1
            self._quarantine(reason="invalid_python", source=source, path=path, digest=raw_digest)
            return None
        counters["python_syntax_valid"] += 1
        for flag in validation.review_flags:
            counters[f"review_flag:{flag}"] += 1
        if has_generated_marker(text, list(self.config["quality"]["generated_markers"])):
            rejections["generated_file"] += 1
            return None
        quality = score_quality(text, validation, self.config["quality"])
        if not quality.accepted:
            rejections[quality.reason or "low_quality"] += 1
            return None
        counters["quality_accepted"] += 1
        relative = path.relative_to(root).as_posix()
        digest = content_sha256(text)
        record = PretrainingRecord(
            record_id=stable_record_id(source.id, relative, digest),
            text=text,
            language="python",
            source_id=source.id,
            source_url=source.official_url,
            repository=source.repository,
            revision=source.version,
            file_path=relative,
            licence_spdx=source.dataset_level_licence,
            content_sha256=digest,
            generation_method="human_or_unknown",
            quality=QualityInfo(True, True, True, quality.score),
            split_group=source.repository or source.id,
        )
        record.validate()
        return record

    def _split_records(self, near: NearDeduplicator) -> Iterator[PretrainingRecord]:
        ratios = {
            name: float(value) for name, value in self.config["splits"]["pretraining"].items()
        }
        yield from split_pretraining_records(near.iter_winners(), ratios, self.seed)

    def build(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Run or resume the complete Phase 2 smoke pipeline."""
        started = time.perf_counter()
        tracemalloc.start()
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        state_path = self.paths["work"] / "pipeline_state.json"
        report_path = self.paths["reports"] / "dataset_report.json"
        if self.resume and state_path.exists() and report_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("completed") and state.get("configuration_sha256") == self.config_hash:
                loaded_report: object = json.loads(report_path.read_text(encoding="utf-8"))
                if not isinstance(loaded_report, dict):
                    raise ValueError("stored dataset report must be an object")
                resumed_report: dict[str, Any] = loaded_report
                resumed_report["resumed_from_completed_run"] = True
                tracemalloc.stop()
                return resumed_report

        sources_path = _resolve(str(self.config["sources_config"]), self.project_root)
        licences_path = _resolve(str(self.config["licences_config"]), self.project_root)
        registry = SourceRegistry.from_yaml(sources_path)
        policy = LicencePolicy.from_yaml(licences_path)
        opt_out_value = self.config.get("opt_out_config")
        opt_out: dict[str, Any] = {}
        if opt_out_value:
            opt_out_path = _resolve(str(opt_out_value), self.project_root)
            loaded_opt_out = yaml.safe_load(opt_out_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_opt_out, dict):
                raise ValueError("opt-out configuration must be a mapping")
            opt_out = loaded_opt_out
        excluded_source_ids = {str(item) for item in opt_out.get("source_ids", [])}
        excluded_repositories = {str(item) for item in opt_out.get("repositories", [])}
        selected_ids = (
            self.source_ids or set(self.config["ingestion"].get("source_ids", [])) or None
        )
        statuses = set(self.config["ingestion"]["approved_statuses"])
        selected = list(registry.iter_sources(selected_ids, statuses))
        selected = [
            source
            for source in selected
            if source.id not in excluded_source_ids
            and (source.repository is None or source.repository not in excluded_repositories)
        ]
        if not selected:
            raise ValueError("no approved sources selected")
        audit = registry.audit(set(policy.allowlist))
        if dry_run:
            tracemalloc.stop()
            return {
                "dry_run": True,
                "source_audit": audit,
                "selected_sources": [s.id for s in selected],
            }

        work = self.paths["work"]
        manifests = self.paths["manifests"]
        exact_db = work / "exact.sqlite3"
        near_db = work / "near.sqlite3"
        leakage_db = work / "leakage.sqlite3"
        for database in (exact_db, near_db, leakage_db):
            _safe_reset_database(database)
        for manifest in (
            manifests / "exact_duplicates.jsonl",
            manifests / "near_duplicates.jsonl",
            self.paths["quarantine"] / "quarantine_manifest.jsonl",
        ):
            if manifest.exists():
                manifest.unlink()

        exact = ExactDeduplicator(exact_db, manifests / "exact_duplicates.jsonl")
        counters: Counter[str] = Counter()
        rejections: Counter[str] = Counter()
        download_results: list[DownloadResult] = []
        failed_sources: list[dict[str, str]] = []
        global_limit = self.maximum_records or int(self.config["ingestion"]["global_record_limit"])
        source_limit = int(self.config["ingestion"]["source_record_limit"])
        try:
            for source in selected:
                if counters["downloaded"] >= global_limit:
                    break
                try:
                    result = prepare_source(
                        source,
                        self.paths["raw"],
                        maximum_download_bytes=self.maximum_download_bytes
                        or int(self.config["ingestion"]["maximum_download_bytes"]),
                        minimum_free_disk_bytes=int(
                            self.config["ingestion"]["minimum_free_disk_bytes"]
                        ),
                    )
                    download_results.append(result)
                    root = Path(result.extracted_path)
                    remaining = global_limit - counters["downloaded"]
                    limit = min(source_limit, remaining)
                    excluded = {
                        str(item).lower() for item in self.config["path_filters"]["excluded_parts"]
                    }
                    for path in iter_source_files(
                        root,
                        source.include_globs,
                        excluded,
                        limit,
                        exclude_globs=source.exclude_globs,
                    ):
                        record = self._process_file(
                            source, root, path, policy, counters, rejections
                        )
                        if record is not None:
                            exact.add(record)
                except Exception as error:
                    LOGGER.exception("source failed", extra={"source_id": source.id})
                    failed_sources.append({"source_id": source.id, "error": type(error).__name__})
        finally:
            counters["exact_deduplicated"] = counters["quality_accepted"] - exact.duplicates

        dedup = self.config["deduplication"]
        near = NearDeduplicator(
            near_db,
            manifests / "near_duplicates.jsonl",
            threshold=float(dedup["near_similarity_threshold"]),
            shingle_size=int(dedup["shingle_size"]),
            permutations=int(dedup["minhash_permutations"]),
            bands=int(dedup["lsh_bands"]),
        )
        try:
            for record in exact.iter_winners():
                near.add(record)
            counters["near_deduplicated"] = counters["exact_deduplicated"] - near.duplicates
            sharding = self.config["sharding"]
            write_shards(
                (record.to_dict() for record in self._split_records(near)),
                self.paths["cleaned"] / "pretraining",
                maximum_uncompressed_bytes=int(sharding["maximum_uncompressed_bytes"]),
                compression_level=int(sharding["compression_level"]),
                config_hash=self.config_hash,
            )
            split_counts: Counter[str] = Counter()
            shard_manifests: list[dict[str, Any]] = []
            for split_name in ("train", "validation", "test"):

                def selected_records(name: str = split_name) -> Iterator[dict[str, Any]]:
                    for item in self._split_records(near):
                        if item.split == name:
                            split_counts[name] += 1
                            yield item.to_dict()

                shard_manifests.extend(
                    write_shards(
                        selected_records(),
                        self.paths["splits"] / "pretraining" / split_name,
                        maximum_uncompressed_bytes=int(sharding["maximum_uncompressed_bytes"]),
                        compression_level=int(sharding["compression_level"]),
                        config_hash=self.config_hash,
                    )
                )
            for split_name in ("train", "validation", "test"):
                (self.paths["splits"] / "instruction" / split_name).mkdir(
                    parents=True, exist_ok=True
                )
            leakage = leakage_report(
                self._split_records(near),
                leakage_db,
                near_threshold=float(dedup["near_similarity_threshold"]),
            )
            counters["split_and_sharded"] = sum(split_counts.values())
            total_bytes = total_chars = total_lines = lexical_tokens = 0
            licences: Counter[str] = Counter()
            source_counts: Counter[str] = Counter()
            for record in near.iter_winners():
                encoded = record.text.encode("utf-8")
                total_bytes += len(encoded)
                total_chars += len(record.text)
                total_lines += len(record.text.splitlines())
                lexical_tokens += validate_python(record.text).lexical_tokens
                licences[record.licence_spdx] += 1
                source_counts[record.source_id] += 1
        finally:
            exact.close()
            near.close()

        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.perf_counter() - started
        downloaded_bytes = sum(result.downloaded_bytes for result in download_results)
        extracted_bytes = sum(result.extracted_bytes for result in download_results)
        compressed_bytes = sum(int(item["compressed_bytes"]) for item in shard_manifests)
        rough_ratio = float(self.config["reporting"]["rough_subword_bytes_per_token"])
        report: dict[str, Any] = {
            "report_scope": "bounded_real_smoke_sample",
            "corpus_version": self.config["reporting"]["corpus_version"],
            "pipeline_configuration_sha256": self.config_hash,
            "cpu_only": bool(self.config.get("cpu_only", True)),
            "source_audit": audit,
            "source_revisions": {source.id: source.version for source in selected},
            "source_distribution": dict(sorted(source_counts.items())),
            "licence_distribution": dict(sorted(licences.items())),
            "downloaded_bytes": downloaded_bytes,
            "extracted_bytes": extracted_bytes,
            "cleaned_bytes": compressed_bytes,
            "total_records": counters["split_and_sharded"],
            "total_python_files": counters["downloaded"],
            "valid_python_percentage": round(
                100 * counters["python_syntax_valid"] / max(1, counters["downloaded"]), 2
            ),
            "utf8_bytes": total_bytes,
            "characters": total_chars,
            "lines": total_lines,
            "python_lexical_tokens": lexical_tokens,
            "rough_subword_token_estimate": round(total_bytes / rough_ratio),
            "rough_token_estimate_note": (
                "UTF-8 bytes divided by configured heuristic; not GenPy tokens"
            ),
            "peak_pipeline_memory_bytes": peak_memory,
            "processing_seconds": round(elapsed, 3),
            "funnel": dict(counters),
            "rejections": dict(sorted(rejections.items())),
            "secret_rejections": rejections["secret_detected"],
            "pii_rejections": rejections["pii_detected"],
            "safety_rejections": rejections["unsafe_content"],
            "static_review_flags": {
                name.removeprefix("review_flag:"): count
                for name, count in sorted(counters.items())
                if name.startswith("review_flag:")
            },
            "scanner_false_positive_control": scanner_false_positive_control(),
            "quality_rejections": rejections["low_quality"] + rejections["minified_content"],
            "exact_duplicate_count": exact.duplicates,
            "near_duplicate_count": near.duplicates,
            "near_duplicate_cluster_sizes": dict(sorted(near.cluster_sizes.items())),
            "split_counts": dict(split_counts),
            "instruction_category_distribution": {},
            "instruction_difficulty_distribution": {},
            "leakage": leakage,
            "shards": shard_manifests,
            "failed_sources": failed_sources,
            "resumed_from_completed_run": False,
        }
        write_dataset_report(report, self.paths["reports"])
        temporary_state = state_path.with_suffix(".json.tmp")
        temporary_state.write_text(
            json.dumps(
                {"completed": True, "configuration_sha256": self.config_hash},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary_state.replace(state_path)
        return report
