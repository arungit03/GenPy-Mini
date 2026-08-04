# Roadmap

## Phase 1 - Planning and Setup

Status: complete.

Define scope, metrics, repository structure, configuration placeholders, environment
requirements, and basic tests. Do not collect data, train a tokenizer, implement the
Transformer, or train a model.

## Phase 2 - Dataset Preparation

Status: pipeline complete; corpus incomplete.

The bounded-memory ingestion, licence filtering, cleaning, static safety scanning, Python
validation, exact/near deduplication, group-aware splitting, Zstandard sharding, and
reporting pipeline is implemented and verified with local fixtures and a pinned real sample.
The GenPy-5M, GenPy-25M, and GenPy-100M corpus tiers have not been built. Large-source
selection and download require explicit approval.

## Phase 3 - Custom Tokenizer

Status: infrastructure complete; smoke tokenizer complete; production tokenizer blocked.

The CPU pipeline now prepares a deterministic train-only corpus, trains GenPy's byte-level BPE
from an empty state, validates and fingerprints artifacts, audits vocabulary, evaluates fixed
and split populations, and streams exact token counts. The 1,024-entry smoke artifact passes.
The current 493,175-byte one-source corpus is below candidate and production thresholds and has
no instruction or held-out records, so the 16,384-token tokenizer was not trained or frozen.

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
