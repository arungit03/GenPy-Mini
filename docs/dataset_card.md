# GenPy Dataset Card

## Summary

Corpus version: `genpy-phase2-expansion-v1`.

This dataset is intended to train GenPy from random weights to generate readable
beginner-to-intermediate Python from natural-language requests. Phase 2 now provides a
verified 12-source pretraining corpus and a deterministically paired instruction corpus, both
built through the same production-oriented pipeline. Neither meets the tokenizer's candidate
(100 MiB) or production (500 MiB) byte thresholds yet — see Scale below.

## Sources And Licences

Twelve pretraining sources are approved: Click 8.1.8, pandas, SciPy, scikit-learn, pytest,
Pydantic, FastAPI, SQLAlchemy, Scrapy, aiohttp, NetworkX, and NLTK, each pinned to an immutable
commit with a verified archive SHA-256 in `configs/data/sources.yaml` and `data/manifests/
source_manifest.jsonl`. Licences are BSD-3-Clause, MIT, or Apache-2.0 per source (see
`docs/data_sources.md` for the full table, per-source nested-licence findings, and the
resulting `exclude_globs`). One instruction source is approved with a distinct
"instruction-only" status that keeps it out of pretraining ingestion: the Exercism Python
track (MIT), paired deterministically by exercise slug. The Stack v2.1.0 remains a recorded
candidate, excluded pending review. Every future source must satisfy the provisional allowlist
and provenance policy in `docs/dataset_policy.md`.

## Scale

The approved pretraining corpus is 4,560 deduplicated records (34,685,551 UTF-8 bytes), split
4,304/256/0 train/validation/test by repository group (98/1/1 ratio; with only 12 groups none
hashed into the 1% test bucket for this seed). The approved instruction corpus is 114 records
from 140 Exercism exercises, split 93/13/8 train/validation/test by problem family (80/10/10
ratio). Combined pretraining-train + instruction-train serialized bytes are 31,818,723 — short
of the tokenizer's 100 MiB candidate threshold and far short of its 500 MiB production
threshold (`configs/tokenizer/genpy_bpe_16k.yaml`). Reaching either threshold requires sources
beyond the current "safer candidate" set; see `docs/data_sources.md` for the larger frameworks
already audited but not yet approved.

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
estimates and predate the 2026-08-04 12-source expansion above; they still reflect only the
original Click-only sample, not the current corpus. Final GenPy token counts require the frozen
16,384-entry production tokenizer, which remains untrained (see Scale above for why: available
bytes are below both the candidate and production readiness thresholds).

## Phase 4 Packed-Data Status

No Phase 2 corpus data has been packed. Phase 4 packed only seven original safe test-fixture
records into seven 64-input-token smoke samples across six isolated family/split shards. The
1,358-byte binary payload and its checksums validate format and loader correctness, not corpus
readiness. Production packing remains blocked by the missing frozen tokenizer.

## Biases, Risks, And Limitations

Repository code overrepresents public open-source conventions and English documentation.
Licence detection, generated-code heuristics, secret/PII scanning, unsafe-content filters,
quality scoring, and near deduplication can produce both false positives and false negatives —
for example, the secret/PII scanners' conservative numeric-sequence and high-entropy-string
heuristics account for most Exercism instruction rejections (e.g. Luhn/ISBN/phone-number
exercises whose sample digit sequences resemble phone numbers). Static validation does not
establish correctness or safety. The 12-source pretraining corpus and single instruction
source are not representative of GenPy V1's full task distribution, and SQLAlchemy's entire
256-record contribution landed in the validation split by the deterministic group hash (an
expected but noteworthy concentration when the underlying record count is small). The dataset
is not guaranteed risk-free or legally safe.

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
python scripts/data/build_dataset.py --config configs/data/phase2.yaml --mode full --confirm-large-download
python scripts/data/build_instruction_dataset.py --config configs/data/phase2.yaml
python scripts/data/validate_dataset.py --config configs/data/phase2.yaml
python scripts/tokenizer/check_readiness.py --config configs/tokenizer/genpy_bpe_16k.yaml
```

The generated report records the exact configuration hash, source revision, archive checksum,
shard checksums, funnel counts, split counts, leakage checks, memory, and processing time. The
`--mode full --confirm-large-download` flags are required beyond the single-source smoke
config because `configs/data/phase2.yaml` now selects all 12 approved pretraining sources; the
Exercism instruction adapter runs separately because its source status is intentionally outside
`ingestion.approved_statuses`, so the main build can never select it.
