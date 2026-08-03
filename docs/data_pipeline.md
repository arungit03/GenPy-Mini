# Data Pipeline

## Flow

```text
Source audit and opt-out check
-> bounded/resumable download or local stream
-> safe archive extraction
-> licence and provenance gate
-> UTF-8 normalization and path filters
-> secret, PII, and unsafe-content scans
-> Python 3.11 AST/tokenize validation
-> quality scoring
-> SQLite exact deduplication
-> shingle MinHash/LSH near deduplication
-> seeded group-aware splits
-> Zstandard JSONL shards, checksums, leakage report, and statistics
```

Downloaded code is never imported or executed. The pipeline uses iterators for files and
shards, while exact hashes, LSH buckets, records, and leakage state live in SQLite. This keeps
memory bounded as the corpus grows. Completed archives and shards use temporary paths and
atomic renames; download and pipeline state allow safe resume.

## Commands

```powershell
python scripts/data/audit_sources.py --config configs/data/sources.yaml
python scripts/data/build_dataset.py --config configs/data/phase2.yaml --mode smoke
python scripts/data/validate_dataset.py --config configs/data/phase2.yaml
python scripts/data/create_splits.py --config configs/data/phase2.yaml
python scripts/data/generate_dataset_report.py --config configs/data/phase2.yaml
```

The build also supports `--dry-run`, `--maximum-records`, `--maximum-download-size`, repeated
`--source`, `--seed`, `--output-directory`, `--workers`, and `--resume`/`--no-resume`.
`--mode full` additionally requires `--confirm-large-download`; that flag is only appropriate
after the user approves source, licence, attribution, size, and disk estimates.

All paths are configuration-driven and use `pathlib`, including Kaggle-compatible input and
working locations. CPU-only operation is the default. Credentials are never required by the
smoke run and must never be placed in configuration.

## Output

Cleaned and split records are compressed `JSONL.zst`. Each shard has record and byte counts,
SHA-256, source/licence composition, UTC creation time, and pipeline configuration hash.
Dataset content and local reports are ignored by Git. Safe source/licence manifests remain
tracked for auditability.
