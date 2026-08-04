# GenPy Tokenizer Card

## Identity And Status

- Name: `genpy-byte-bpe-smoke-1k`
- Version: 1
- Algorithm: custom byte-level BPE from an empty vocabulary state
- Vocabulary: 1,024 entries including seven special tokens
- Artifact: `artifacts/tokenizer/smoke`
- Freeze status: smoke, not frozen, prohibited for GenPy model training
- Tokenizer fingerprint: `29a6b2770f043dc2ef0732a81c0b524b6e3ee678c54d0d10b95cbf294678b31f`
- Corpus fingerprint: `d0b62b9df93aa1fc7a5e1e8a22b2b75956dabd0b188f4188abe03727c3323e2e`

## Intended Use

This artifact verifies Phase 3 corpus preparation, custom tokenizer training, serialization,
evaluation, exact counting, validation, and packaging on CPU. It is not the shared 16,384-token
production tokenizer and must not be used to train GenPy-5M, GenPy-25M, or GenPy-100M.

Phase 4 uses this fingerprint only with `GenPy-Smoke` and original safe packing fixtures. Model
and packed-shard loaders reject fingerprint mismatches. Production model configurations continue
to reference the unpopulated 16,384-token artifact and were not changed to this smoke fingerprint.

## Special Tokens

`<|pad|>`=0, `<|bos|>`=1, `<|eos|>`=2, `<|user|>`=3, `<|assistant|>`=4,
`<|code|>`=5, and `<|end|>`=6.

## Training Data

The deterministic training manifest selected 63 pretraining records and 493,175 serialized
bytes. Actual composition was 100% pretraining and 0% instruction. Every record came from the
pinned Click 8.1.8 source at commit `934813e4d421071a1b3db3973c02fe2721359a6e`
under BSD-3-Clause. Phase 2 applied licence checks, static secrets/PII/safety filtering, Python
validation, exact and near deduplication, and deterministic group splitting. Only train records
were used.

## Evaluation

All 14 fixed valid UTF-8 cases and both fixed instruction cases encoded and round-tripped
exactly. Unknown count was zero; special-token atomicity, indentation, newline, and save/load
checks were 100%. Fixed-suite compression was 1.8851 UTF-8 bytes/token, below the 2.5 goal.
Median identifier fragmentation was 7.5, above the goal of 3. Both fixed V1 instruction samples
fit in 1,024 tokens.

On the full smoke pretraining split, 63 records became 167,002 content tokens plus 252
structural tokens. Median serialized length was 654 and p95 was 9,884; 58.73% of records fit in
1,024 tokens. Validation, test, and all instruction splits are empty.

The vocabulary audit found no suspicious secret/PII patterns, structural errors, or invalid
merge references. This static result does not prove the absence of sensitive fragments.

## Biases, Risks, And Limitations

The corpus is one English-oriented Python library and is not representative of beginner and
intermediate prompt-to-code tasks. It contains no production instruction data or held-out
records. Quality and performance measurements are small and machine-dependent. Byte coverage
does not establish useful segmentation or downstream model quality. Decoded code must never be
executed outside an isolated sandbox.

## Reproduction

```powershell
python scripts/tokenizer/check_readiness.py --config configs/tokenizer/smoke_tokenizer.yaml
python scripts/tokenizer/prepare_corpus.py --config configs/tokenizer/smoke_tokenizer.yaml --mode smoke
python scripts/tokenizer/train_tokenizer.py --config configs/tokenizer/smoke_tokenizer.yaml --mode smoke
python scripts/tokenizer/validate_tokenizer.py --artifact artifacts/tokenizer/smoke
python scripts/tokenizer/evaluate_tokenizer.py --config configs/tokenizer/evaluation.yaml
python scripts/tokenizer/count_corpus_tokens.py --config configs/tokenizer/smoke_tokenizer.yaml --artifact artifacts/tokenizer/smoke --resume
python scripts/tokenizer/package_tokenizer.py --artifact artifacts/tokenizer/smoke
```
