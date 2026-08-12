# GenPy-200M — STEP 3 TODO

States: `[ ]` Not started, `[~]` In progress, `[x]` Completed and verified, `[!]` Blocked.

## TODO 1 — Audit Existing Repository
- [x] Inspect Step 1/2, README, TODOs, docs, configs, tokenizer scaffolding, and run the existing test suite. Baseline: 32 passed.

## TODO 2 — Create STEP3_TODO.md
- [x] Create this checklist while preserving `STEP1_TODO.md` and `STEP2_TODO.md`.

## TODO 3 — Extend Project Structure
- [x] Add tokenizer config, corpus/builder/tokenizer/trainer/evaluation/validation modules, CLIs, fixture, tests, and documentation without model code.

## TODO 4 — Create Tokenizer Configuration
- [x] Add exact Byte-Level BPE 32K, NFC, ByteLevel, special-token, corpus, output, and production-gate configuration.

## TODO 5 — Extend Configuration Loader
- [x] Add validated tokenizer dataclasses while preserving all prior config classes; require production vocab 32000, byte_level_bpe, NFC, and unique four special tokens.

## TODO 6 — Verify Architecture/Tokenizer Contract
- [x] Reusable validation must require model and tokenizer vocabulary sizes both equal 32000.

## TODO 7 — Implement Corpus Reader
- [x] Stream deterministic train/validation JSONL.GZ shards, extract text, track docs/chars/bytes, support document/byte limits, and report malformed/missing data.

## TODO 8 — Connect Step 2 Manifest Provenance
- [x] Deterministically locate or explicitly select a completed Step 2 manifest and record source/config/cleaning/shards without copying unnecessary data.

## TODO 9 — Implement Tokenizer Builder
- [x] Build standalone `tokenizers` Byte-Level BPE with NFC, ByteLevel prefix-space false/regex true, ByteLevel alphabet, BPE min frequency 2/max length 64, and exactly four special tokens in IDs 0–3. No pretrained tokenizer/transformers.

## TODO 10 — Implement Tokenizer Training
- [x] Train from an iterator without collecting the corpus, report docs/chars/bytes, and make no GPU claims.

## TODO 11 — Implement Production Corpus Gate
- [x] Classify synthetic/development/production candidate using real train bytes; production minimum 256 MiB and target approximately 1 GiB; current repository has no real shards.

## TODO 12 — Implement Smoke Training Mode
- [x] Add fast synthetic 512-vocabulary smoke training into a temporary/non-production directory with clear labeling.

## TODO 13 — Implement Production Training CLI
- [x] Add `train_tokenizer.py` options for config/production/smoke/input/output/limits/manifest/force, summary-before-training, and no overwrite without `--force`; production execution correctly stops without real Step 2 provenance.

## TODO 14 — Verify Vocabulary Size
- [x] Implemented exact vocabulary validation and verified it rejects non-expected sizes; production 32000 training remains pending the real corpus.

## TODO 15 — Verify Special Token IDs
- [x] Validate `<|pad|>=0`, `<|bos|>=1`, `<|eos|>=2`, `<|unk|>=3` without remapping hacks.

## TODO 16 — Implement GenPyTokenizer Wrapper
- [x] Add encode/decode and vocab/special-ID properties; no automatic padding, truncation, BOS, or EOS.

## TODO 17 — Add Explicit Document Boundary Helper
- [x] Add `encode_document` that appends EOS only and does not add BOS or sequence packing.

## TODO 18 — Implement Save Functionality
- [x] Save canonical tokenizer JSON and config/manifest artifacts; evaluation artifact support is implemented by the evaluation CLI.

## TODO 19 — Implement Load Functionality
- [x] Reload from tokenizer JSON without a corpus and guarantee identical encoding.

## TODO 20 — Generate Tokenizer Manifest
- [x] Record architecture, special IDs, settings, corpus provenance/counts, mode, versions, timestamp, and tokenizer checksum without secrets.

## TODO 21 — Implement Artifact Checksum
- [x] Compute and verify SHA-256 of canonical tokenizer JSON.

## TODO 22 — Implement Tokenizer Validation Module
- [x] Validate file/JSON/vocab/special IDs/architecture match/ByteLevel/NFC/encode/decode/save-load/checksum with clear errors.

## TODO 23 — Implement Round-Trip Tests
- [x] Test NFC-normalized English, numbers, punctuation, spaces, newlines, code, URLs, mathematics, accented Latin, Tamil, Hindi, Chinese, emoji, and mixed text.

## TODO 24 — Verify No Unknown Tokens for Valid Unicode Samples
- [x] Assert zero UNK for representative valid ASCII, accented, Tamil, Devanagari, CJK, emoji, math, and programming text.

## TODO 25 — Verify Whitespace Preservation
- [x] Test leading/internal spaces, newlines, tabs, paragraphs, and expected normalized round-trips.

## TODO 26 — Verify add_prefix_space=False Behavior
- [x] Verify beginning `Hello` does not decode with an unwanted leading space.

## TODO 27 — Test Explicit BOS/EOS Behavior
- [x] Verify raw encode adds neither token and wrapper flags add them only when requested.

## TODO 28 — Implement Tokenizer Evaluation
- [x] Evaluate validation shards preferentially with docs/chars/bytes/tokens/UNK and ratio metrics; no validation shards currently exist and the CLI reports this explicitly.

## TODO 29 — Evaluate by Content Type
- [x] Add original diagnostic English/code/C/numbers/math/URLs/Tamil/Hindi/emoji/mixed-language suite with token counts.

## TODO 30 — Implement inspect_tokenizer.py
- [x] Show safe original text, IDs, token strings/count, decoded text, and round-trip status with Unicode-safe output.

## TODO 31 — Implement evaluate_tokenizer.py
- [x] Support tokenizer/input/max-documents/output and prefer validation shards; write JSON metrics, with an explicit missing-corpus error.

## TODO 32 — Implement verify_tokenizer.py
- [x] Verify all artifact invariants and print PASS only after actual checks; production invocation remains gated by real corpus availability.

## TODO 33 — Create Tokenizer Configuration Tests
- [x] Test loading, 32000 vocab, tokens/uniqueness, invalid vocab/frequency, and model mismatch.

## TODO 34 — Create Corpus Reader Tests
- [x] Test compressed reading/order/text/malformed/missing/limits/train-only/validation-only with temporary data and no internet.

## TODO 35 — Create Tokenizer Training Tests
- [x] Train a fast small tokenizer, test vocabulary/special IDs/ByteLevel/save/reload.

## TODO 36 — Create Encoding Tests
- [x] Test English/Unicode/Tamil/emoji/whitespace/newlines/code/BOS/EOS/no implicit tokens/prefix/determinism.

## TODO 37 — Create Evaluation Tests
- [x] Test metric arithmetic for tokens/chars/bytes/ratios/UNK.

## TODO 38 — Create Validation Tests
- [x] Detect wrong vocab/IDs/missing or corrupt files/manifest/checksum/model mismatch.

## TODO 39 — Run Smoke Tokenizer
- [x] Run `python scripts/train_tokenizer.py --smoke`; verify training/save/reload/round-trip/special IDs.

## TODO 40 — Run Full Offline Tests
- [x] Run all Step 1/2/3 tests with zero failures: 45 passed.

## TODO 41 — Inspect Real Step 2 Corpus
- [x] Inspect train/validation shards and manifest provenance; no real train/validation shards or completed manifest are present.

## TODO 42 — Handle Missing Real Corpus Correctly
- [x] Real corpus is unavailable; no final 32K tokenizer was trained or claimed, and the blocker/recommended Kaggle preparation are documented.

## TODO 43 — Production Corpus Preparation Gate
- [x] Production gate is implemented and requires Step 2-cleaned real train text >=256 MiB with an approximately 1 GiB target; current repository has no qualifying corpus.

## TODO 44 — Train Production GenPy Tokenizer
- [!] Blocked until genuine real Step 2 cleaned corpus is available; never label synthetic training production.

## TODO 45 — Validate Production Vocabulary
- [!] Blocked with TODO 44; production 32000 vocabulary and IDs cannot be claimed without real-corpus training.

## TODO 46 — Run Production Round-Trip Suite
- [!] Blocked with TODO 44; run only against a genuine production tokenizer.

## TODO 47 — Evaluate on Validation Corpus
- [!] Blocked because real Step 2 validation shards are unavailable.

## TODO 48 — Validate Saved Production Artifact
- [!] Blocked because no production artifact may be fabricated.

## TODO 49 — Test Clean Reload
- [!] Blocked for production artifact; smoke reload is covered separately.

## TODO 50 — Update Documentation
- [x] Add `docs/TOKENIZER.md` documenting implementation, smoke/development/production distinction, corpus, settings, boundaries, evaluation, and limitations.

## TODO 51 — Update .gitignore
- [x] Keep tokenizer artifacts ignored while preserving config/docs/tests/source.

## TODO 52 — Scope Audit
- [x] Confirm no Step 4+ model, attention, training, packing, generation, or pretraining implementation.

## TODO 53 — Forbidden Dependency Audit
- [x] Confirm no GPT2/Llama/AutoTokenizer/transformers/pretrained tokenizer usage and no transformers dependency.

## TODO 54 — Final Regression Test
- [x] Run pytest, environment checker, and Step 2 synthetic validation; environment remains CPU-only and Step 2 tests pass.

## TODO 55 — Final Production Verification
- [!] Blocked until real production tokenizer exists; do not claim PASS.

## TODO 56 — Update README
- [x] Keep Step 3 `[~]` while production corpus is unavailable and link tokenizer documentation.

## TODO 57 — Final TODO Audit
- [x] Read this file and report implementation completion separately from production completion, with blockers explicit: implementation complete; production tokenizer blocked by missing real corpus.

## Tokenizer Quality Rules

No pretrained tokenizer, fake 32K vocabulary, ASCII conversion, lowercasing, implicit BOS/EOS/padding, sequence packing, validation-data training, Transformer/model/training implementation, or fabricated production result.
