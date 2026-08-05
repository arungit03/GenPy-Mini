# Roadmap

## Phase 1 - Planning and Setup

Status: complete.

Define scope, metrics, repository structure, configuration placeholders, environment
requirements, and basic tests. Do not collect data, train a tokenizer, implement the
Transformer, or train a model.

## Phase 2 - Dataset Preparation

Status: pipeline complete; 12-source pretraining + Exercism instruction corpus approved and
ingested (2026-08-04); still below tokenizer-training scale.

The bounded-memory ingestion, licence filtering, cleaning, static safety scanning, Python
validation, exact/near deduplication, group-aware splitting, Zstandard sharding, and
reporting pipeline is implemented and verified with local fixtures and now a 12-source real
expansion: Click, pandas, SciPy, scikit-learn, pytest, Pydantic, FastAPI, SQLAlchemy, Scrapy,
aiohttp, NetworkX, and NLTK, each pinned, checksum-verified, and nested-licence reviewed (see
`docs/data_sources.md`). The approved pretraining corpus is 4,560 deduplicated records
(34,685,551 UTF-8 bytes); the approved Exercism instruction corpus (deterministic
instructions.md/example.py pairing, MIT, instruction-only source status) is 114 records from
140 exercises. Combined available tokenizer-training bytes (31,818,723) remain below both the
100 MiB candidate and 500 MiB production thresholds, so the GenPy-5M/25M/100M corpus tiers are
still not built. Reaching production scale requires sources beyond the current "safer
candidate" set — see the larger-framework audit table in `docs/data_sources.md`.

## Phase 3 - Custom Tokenizer

Status: infrastructure complete; smoke tokenizer complete; production tokenizer blocked.

The CPU pipeline now prepares a deterministic train-only corpus, trains GenPy's byte-level BPE
from an empty state, validates and fingerprints artifacts, audits vocabulary, evaluates fixed
and split populations, and streams exact token counts. The 1,024-entry smoke artifact passes.
Following the 2026-08-04 Phase 2 expansion, the corpus now has instruction records and a
non-empty held-out instruction split, but combined pretraining-train + instruction-train
serialized bytes (31,818,723) remain below the 100 MiB candidate threshold and far below the
500 MiB production threshold
(`python scripts/tokenizer/check_readiness.py --config configs/tokenizer/genpy_bpe_16k.yaml`
reports `READY_FOR_SMOKE_TOKENIZER`), so the 16,384-token tokenizer was still not trained or
frozen.

## Phase 4 - Model Implementation

Status: infrastructure and smoke integration complete; production packing blocked.

The configuration-driven Transformer, exact parameter audit, compatibility contract,
deterministic packer, checksummed binary format, memory-mapped loader, sampler, CPU
forward/backward, and bounded safe-fixture overfit are verified. Production data was not packed
because the production tokenizer is not frozen.

## Phase 5 - Training from Random Weights

Status: next but not ready. Implement a resource-budgeted training engine, then pretrain GenPy-5M
on approved packed data before considering GenPy-25M or GenPy-100M.

## Phase 6 - Evaluation and Improvement

Evaluate on the frozen private evaluation set, analyze failures, and improve data,
training, and decoding.

## Phase 7 - Application and Release

Package local inference, safety controls, documentation, and release artifacts.
