# GenPy Dataset Card

## Summary

Corpus version: `genpy-phase2-smoke-v1`.

This dataset is intended to train GenPy from random weights to generate readable
beginner-to-intermediate Python from natural-language requests. Phase 2 currently provides a
verified bounded real-source sample and a production-oriented pipeline, not a training-scale
corpus.

## Sources And Licences

The current real smoke source is Click 8.1.8, pinned to commit
`934813e4d421071a1b3db3973c02fe2721359a6e` under BSD-3-Clause. The Stack v2.1.0 is recorded as
a candidate but excluded pending explicit review. Every future source must satisfy the
provisional allowlist and provenance policy in `docs/dataset_policy.md`.

## Collection And Processing

Sources are streamed or downloaded with byte limits, disk checks, checksums, resume state,
and atomic completion. Python is decoded as UTF-8, normalized without reformatting, statically
scanned, parsed as Python 3.11, scored, globally exact/near deduplicated, and assigned by
repository/project group. Pretraining ratios are 98/1/1; instruction ratios are 80/10/10.
Outputs are deterministic Zstandard JSONL shards with manifests.

Instruction records use system/user/assistant messages with exactly one syntactically valid
assistant code answer. No production instruction corpus has been generated, and no LLM was
used to generate one. Test-only examples remain under `tests/fixtures/`.

## Intended Use

The eventual corpus is for GenPy tokenizer training, pretraining, and instruction tuning.
It is not intended for identity inference, personal-data processing, malware development,
legal compliance claims, or direct execution of untrusted code.

## Smoke Tokenizer Counts

The Phase 3 smoke tokenizer fingerprint is
`29a6b2770f043dc2ef0732a81c0b524b6e3ee678c54d0d10b95cbf294678b31f`.
It is a 1,024-entry infrastructure artifact, not the production GenPy tokenizer. Under this
artifact, pretraining train has 63 records, 491,222 UTF-8 source bytes, 167,002 content tokens,
252 structural tokens, and 167,254 total serialized tokens. Median serialized length is 654,
p90 is 6,537, p95 is 9,884, and p99 is 34,530. Twenty-six records exceed 1,024 tokens and
58.73% fit. Pretraining validation/test and all instruction splits contain zero records.

These exact smoke-tokenizer counts supplement rather than replace Phase 2's historical rough
estimates. Final GenPy token counts require the frozen 16,384-entry production tokenizer.

## Phase 4 Packed-Data Status

No Phase 2 corpus data has been packed. Phase 4 packed only seven original safe test-fixture
records into seven 64-input-token smoke samples across six isolated family/split shards. The
1,358-byte binary payload and its checksums validate format and loader correctness, not corpus
readiness. Production packing remains blocked by the missing frozen tokenizer.

## Biases, Risks, And Limitations

Repository code overrepresents public open-source conventions and English documentation.
Licence detection, generated-code heuristics, secret/PII scanning, unsafe-content filters,
quality scoring, and near deduplication can produce both false positives and false negatives.
Static validation does not establish correctness or safety. The current smoke sample is one
library repository and is not representative of GenPy V1 tasks. The dataset is not guaranteed
risk-free or legally safe.

## Personal Data And Safety

Suspicious records are rejected before cleaned output. Quarantine stores only category,
content hash, source ID, and path. No discovered value is printed or retained in reports.
Scanner performance must be reviewed on safe controls and sampled source findings before each
large build. Downloaded code is never executed.

## Opt-Out And Removal

Validated source or repository removals are added to `configs/data/opt_out.yaml`, affected
artifacts are deleted from working storage, and the corpus is rebuilt under a new version.
Upstream source removal and attribution requirements remain binding. Requests involving a
gated dataset must also follow that dataset's official process.

## Reproduction

```powershell
pip install -r requirements-data.txt
python scripts/data/audit_sources.py --config configs/data/sources.yaml
python scripts/data/build_dataset.py --config configs/data/phase2.yaml --mode smoke
python scripts/data/validate_dataset.py --config configs/data/phase2.yaml
python scripts/data/generate_dataset_report.py --config configs/data/phase2.yaml
```

The generated report records the exact configuration hash, source revision, archive checksum,
shard checksums, funnel counts, split counts, leakage checks, memory, and processing time.
