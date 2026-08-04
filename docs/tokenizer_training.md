# Tokenizer Training

## Readiness

Production training is allowed only when Phase 2 manifests, licence evidence, checksums,
deduplication, leakage checks, train-only stable records, representative instruction data,
held-out splits, and the configured 500 MiB minimum all pass. Candidate status begins at 100
MiB. A smaller verified corpus may train only an isolated smoke tokenizer.

The current gate returns `READY_FOR_SMOKE_TOKENIZER`: 63 pretraining records and 493,175
serialized bytes are available; instruction, validation, and test records are absent.

## Corpus Construction

`prepare_corpus.py` streams checksum-verified Zstandard JSONL shards in family, shard-name,
and record order. It accepts only `train` records, validates schema IDs and content hashes,
applies byte budgets, and writes a metadata-only manifest. Source text is not copied into the
manifest. The configured starting mixture is 85% pretraining and 15% instruction by bytes;
families are never duplicated to force that ratio, and the actual mixture is reported.

The production sample limit is 1 GiB. The production gate requires at least 500 MiB and the
candidate gate requires 100 MiB. Paths and thresholds live in the YAML configurations.

## Algorithm

Training initializes an empty BPE model with the complete byte alphabet, the locked seven
special tokens, a byte-level pre-tokenizer, and byte-level decoder. Normalization, lowercasing,
prefix-space insertion, unknown-token fallback, and BPE dropout are disabled. Seed 42 and one
worker are recorded. The requested vocabulary size includes special tokens and must be reached
exactly.

Important configuration fields include tokenizer name/version/status, vocabulary size,
minimum frequency, byte-level behavior, seed, artifact path, train-shard paths, byte limits,
mixture, readiness thresholds, and CPU worker settings.

## Commands

Windows PowerShell:

```powershell
python scripts/tokenizer/check_readiness.py --config configs/tokenizer/genpy_bpe_16k.yaml
python scripts/tokenizer/prepare_corpus.py --config configs/tokenizer/smoke_tokenizer.yaml --mode smoke
python scripts/tokenizer/train_tokenizer.py --config configs/tokenizer/smoke_tokenizer.yaml --mode smoke
python scripts/tokenizer/validate_tokenizer.py --artifact artifacts/tokenizer/smoke
python scripts/tokenizer/evaluate_tokenizer.py --config configs/tokenizer/evaluation.yaml
python scripts/tokenizer/count_corpus_tokens.py --config configs/tokenizer/smoke_tokenizer.yaml --artifact artifacts/tokenizer/smoke --resume
python scripts/tokenizer/package_tokenizer.py --artifact artifacts/tokenizer/smoke
```

Linux and Kaggle use the same commands after activating the Python 3.11 environment. The
pipeline is CPU-only, does not require CUDA, and does not require network access.

Use `--dry-run` to inspect readiness, `--max-records` or `--max-bytes` for bounded selection,
`--source` for approved-source filtering, and `--output` for an alternate smoke path. The
versioned seed must match the config. Deterministic training currently requires one worker.

## Resume And Artifacts

Corpus manifests are deterministic and may be regenerated. Exact token counting resumes from
per-shard state only when both shard checksum and tokenizer fingerprint match. BPE training
itself restarts from the manifest; it does not resume a partially trained model.

Smoke files are written under `artifacts/tokenizer/smoke`. The reserved production location is
`artifacts/tokenizer/genpy-byte-bpe-16k-v1`. Generated corpora, reports, caches, and tokenizer
artifacts are ignored by Git.

## Retraining Rules

Smoke output may be replaced only with explicit `--force`. Production output is never
overwritten under the same version, even with `--force`. Corpus, configuration, vocabulary,
merge, or special-token changes require a new version and complete readiness, training,
evaluation, counting, audit, checksum, and model-config updates.

## Troubleshooting

- `READY_FOR_SMOKE_TOKENIZER` on the production config means the data tier is too small or
  incomplete; do not lower the thresholds.
- An actual vocabulary smaller than requested means the corpus cannot support enough merges;
  use a smaller smoke config, never a smaller production config.
- A checksum failure means the shard or artifact is corrupted; rebuild from approved inputs.
- A reserved-token collision must be rejected or quarantined, not escaped ad hoc.
- A vocabulary security finding requires removing the responsible records and retraining a new
  candidate; never edit `vocab.json` or `merges.txt` manually.
