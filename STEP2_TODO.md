# GenPy-200M — STEP 2 TODO

States: `[ ]` Not started, `[~]` In progress, `[x]` Completed and verified, `[!]` Problem/blocker.

## TODO 1 — Audit Existing Step 1 Repository
- [x] Inspect tree, read README/STEP1_TODO/AGENTS when present, run baseline pytest, verify configuration utilities, preserve Step 1, and identify root.

## TODO 2 — Create STEP2_TODO.md
- [x] Create this complete checklist without modifying or deleting STEP1_TODO.md; verify both exist.

## TODO 3 — Extend Project Structure
- [x] Add data config, genpy/data modules, CLI scripts, fixture/tests, documentation, and manifests placeholder; preserve Step 1. Verify files/imports.

## TODO 4 — Create Dataset Configuration
- [x] Add FineWeb-Edu `sample-10BT`, train, streaming, text field, cleaning, split, JSONL.GZ output, sharding, manifests, resume, and metadata settings. Load/validate all sections.

## TODO 5 — Extend Configuration Loader
- [x] Add validated DatasetConfig, ProcessingConfig, DataSplitConfig, OutputConfig, ResumeConfig, MetadataConfig, and DataPipelineConfig without breaking ModelConfig/TrainingConfig. Validate fields, thresholds, fraction, seeds, shard size, and jsonl.gz.

## TODO 6 — Define Standardized Document Schema
- [x] Add compact GenPyDocument with deterministic ID, text/hash/source metadata/quality/count/split fields; optional metadata may be None.

## TODO 7 — Implement Dataset Source Loader
- [x] Add lazy Hugging Face streaming loader, configurable limit, clear access errors, no cleaning, no whole-dataset RAM load, and local test source support.

## TODO 8 — Create Dataset Inspection Script
- [x] Add tiny-limit CLI showing dataset/config/streaming/fields/counts/lengths/metadata/truncated previews without full documents.

## TODO 9 — Implement Text Normalization
- [x] Add conservative NFC, line-ending, control-character, whitespace, trailing-space, and blank-line normalization preserving paragraphs, Unicode, punctuation, symbols, and code.

## TODO 10 — Implement Quality Filters
- [x] Reject missing/non-string/empty/too-short/too-long text with transparent reasons; no classifiers or undocumented heuristics.

## TODO 11 — Implement Exact Deduplication
- [x] SHA-256 normalized UTF-8 exact deduplication, retain first, count later duplicates; no fuzzy/semantic methods.

## TODO 12 — Implement Deterministic Train/Validation Split
- [x] Stable content-hash plus seed assignment, default validation fraction .005; same input/seed stable and seed changes can alter assignment.

## TODO 13 — Implement Processed Shard Writer
- [x] Separate deterministic UTF-8 JSONL.GZ train/validation shards, configurable 25,000 rollover, atomic .tmp finalization, safe close.

## TODO 14 — Implement Dataset Statistics
- [x] Track source/accepted/rejected/duplicates, split counts, chars/bytes, min/max/average lengths, reasons; serialize JSON and no token counts.

## TODO 15 — Implement Manifest System
- [x] Emit JSON manifest with version/time/source/config/stats/shards/completion and optional Python/datasets versions, no secrets.

## TODO 16 — Implement Resume State
- [x] Store counters, shard indexes/files, stats, hashes, fingerprint, completion; check incompatible config, handle .tmp, document streaming skip limitation.

## TODO 17 — Implement Main Data Pipeline
- [x] Orchestrate source → extraction → normalize → filter → hash/dedup → split → schema → writer → stats → manifest/state incrementally with max_documents and restrained progress.

## TODO 18 — Create prepare_data.py
- [x] CLI supports --config, --max-documents, --output-dir, --resume, concise summary, and offline local source.

## TODO 19 — Create validate_data.py
- [x] Validate gzip JSONL, JSON/schema/UTF-8/text/hash/split/counts, duplicate absence, manifests, truncation/corruption, and nonzero failure.

## TODO 20 — Create Local Fixture Dataset
- [x] Add local JSONL valid/duplicate/short/long/Unicode/whitespace/metadata/edge cases; no download.

## TODO 21 — Create Data Configuration Tests
- [x] Test valid load, required fields, invalid thresholds/fraction/shard/seed/output.

## TODO 22 — Create Cleaning Tests
- [x] Test NFC, line endings, controls, whitespace/paragraphs, Unicode/code, and rejection reasons.

## TODO 23 — Create Deduplication Tests
- [x] Test SHA-256 stability, first retention, duplicate detection, and distinct content.

## TODO 24 — Create Writer Tests
- [x] Test gzip/JSONL, rollover, split-separated deterministic names, atomic finalization, and safe close in temporary dirs.

## TODO 25 — Create Pipeline Integration Tests
- [x] Run fixture → cleaning → filter → dedup → split → writing → stats → validation; verify valid/rejected/duplicates/stats/shards/manifest, offline only.

## TODO 26 — Test Resume Behavior
- [x] Simulate interruption/resume; verify no duplication, counters, shard numbering, incompatible rejection, valid final output, and no silent overwrite.

## TODO 27 — Create Dataset Documentation
- [x] Document source/license responsibility, pipeline, every cleaning rule, schema, resume/streaming limits, no tokenization, and data limitations.

## TODO 28 — Update .gitignore Safely
- [x] Ignore processed artifacts/manifests while preserving .gitkeep; raw/tokenizer/checkpoint/log ignores remain; configs/docs/fixtures/STEP2_TODO remain trackable.

## TODO 29 — Run Static Import Verification
- [x] Verify config/schema/cleaning/dedup/split imports and root scripts without fragile path workarounds.

## TODO 30 — Run Complete Offline Test Suite
- [x] Run pytest with all Step 1/2 tests and zero failures.

## TODO 31 — Run Local Synthetic Smoke Test
- [x] Run offline fixture pipeline, validate temporary output, verify shards/manifest/dedup/rejections, clean temporary artifacts.

## TODO 32 — Perform Tiny FineWeb-Edu Connectivity Check
- [x] Attempted the configured 3-document FineWeb-Edu inspection; external access was blocked by the environment's Hugging Face SSL certificate failure, with no substitution or fabricated result.

## TODO 33 — Optional 100–1000 Document Real Smoke Run
- [x] Skipped the optional real-data smoke run because the tiny connectivity check did not succeed; offline verification passed.

## TODO 34 — Inspect Statistics Sanity
- [x] Verify count/split/duplicate/rejection/character/byte invariants and add tests where useful.

## TODO 35 — Inspect Resume Safety
- [x] Review state boundaries, completed shards, fingerprint, incompatible resume, overwrite safety, and temporary-file handling.

## TODO 36 — Scope Violation Audit
- [x] Search for tokenizer/token IDs/packing, Transformer/attention/RoPE/RMSNorm/SwiGLU, optimizer/scheduler/training/checkpoint/generation and pretrained shortcuts; docs mentions allowed.

## TODO 37 — Final Repository Audit
- [x] Inspect tree, preserve Step 1, verify Step 2, check data/checkpoints/logs and no accidental large files.

## TODO 38 — Final Test Suite
- [x] Run pytest, environment checker, and synthetic validation again; Step 1 remains functional.

## TODO 39 — Update README
- [x] After mandatory checks mark Step 2 [x], leave Step 3 [ ], and add a short DATA_PIPELINE.md link.

## TODO 40 — Final TODO Audit
- [x] Read this file fully and calculate Total/Completed/Remaining/Blocked. Required Remaining 0 and Blocked 0. Offline success plus accurately reported network limits is acceptable.

## Data Quality and Scope Rules
- Preserve information conservatively, deterministically, transparently, reproducibly, and incrementally.
- Step 2 remains text-only: no tokenizer, token IDs/counts, model, training, or pretrained shortcuts.
- Never download FineWeb-Edu completely or collect it into RAM.
- Investigate, fix, rerun, and verify every command failure before completion.
