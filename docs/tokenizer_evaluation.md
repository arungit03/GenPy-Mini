# Tokenizer Evaluation

These are actual results from the CPU smoke artifact built on 2026-08-03. They are not
production acceptance results.

## Artifact

| Field | Result |
| --- | --- |
| Name | `genpy-byte-bpe-smoke-1k` |
| Status | smoke, not frozen |
| Requested / actual vocabulary | 1,024 / 1,024 |
| Fingerprint | `29a6b2770f043dc2ef0732a81c0b524b6e3ee678c54d0d10b95cbf294678b31f` |
| Corpus fingerprint | `d0b62b9df93aa1fc7a5e1e8a22b2b75956dabd0b188f4188abe03727c3323e2e` |
| Training records / serialized bytes | 63 / 493,175 |

## Acceptance Checks

| Check | Result |
| --- | ---: |
| Fixed valid UTF-8 cases | 14 |
| Encoding failures / decode failures | 0 / 0 |
| Exact fixed-case round trip | 100.00% |
| Instruction round trip | 100.00% |
| UTF-8 byte coverage | 100.00% |
| Unknown tokens | 0 |
| Special-token atomicity and ID stability | 100.00% |
| Indentation / newline preservation | 100.00% / 100.00% |
| Save/load equivalence | 100.00% |
| Checksum validation | passed |

## Quality And Lengths

| Population | Records | Bytes/token | Median | p90 | p95 | p99 | Fits 1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Python and Unicode | 14 | 1.8851 | 21 | 35 | 3,273 | 3,273 | 92.86% |
| V1 instruction examples | 2 | 3.1096 | 52 | 94 | 94 | 94 | 100.00% |
| Deterministic training sample | 32 | 3.0452 | 509 | 9,884 | 11,950 | 34,530 | 65.63% |
| Phase 2 validation | 0 | n/a | 0 | 0 | 0 | 0 | n/a |
| Phase 2 test | 0 | n/a | 0 | 0 | 0 | 0 | n/a |

Median identifier fragmentation was 7.5 tokens and average Python keyword fragmentation was
2.0625 tokens. The fixed-suite compression and identifier goals were missed. The two V1
instruction examples both fit 1,024 tokens, but this tiny fixed sample is not representative.

## Full Smoke Corpus Counts

Pretraining train contains 63 records, 491,222 source bytes, 167,002 content tokens, 252
structural tokens, and 167,254 serialized tokens. Median length was 654; p90 6,537; p95 9,884;
p99 34,530. Twenty-six records exceeded 1,024 tokens, 58.73% fit, and the simple aggregate
estimate is 164 packed sequences. All other family/split combinations contain zero records.

## Vocabulary Audit

The audit passed with zero suspicious secret/PII findings, zero structural errors, and zero
invalid merge references. It reported 125 tokens that are not reached by re-encoding their
standalone decoded fragment; these remain visible for review and do not represent an acceptance
claim. Static scanning cannot guarantee complete sensitive-data removal.

## Determinism

Two independent trainings in the same Windows environment produced identical `vocab.json`,
`merges.txt`, special IDs, tokenizer fingerprint, and fixed-example encodings. Cross-platform
byte identity has not been tested or claimed.
