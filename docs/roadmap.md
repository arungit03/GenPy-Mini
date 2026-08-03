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

Next phase after corpus approval. Freeze a versioned corpus, then train a byte-level BPE
tokenizer only on GenPy data with a 16,384-token vocabulary and recalculate exact token counts.

## Phase 4 - Model Implementation

Implement the decoder-only Transformer architecture and focused tests.

## Phase 5 - Training from Random Weights

Train GenPy-5M, GenPy-25M, and GenPy-100M from random initialization on Kaggle GPU.

## Phase 6 - Evaluation and Improvement

Evaluate on the frozen private evaluation set, analyze failures, and improve data,
training, and decoding.

## Phase 7 - Application and Release

Package local inference, safety controls, documentation, and release artifacts.
