# GenPy Checkpoint 2.5 Report

## Status

COMPLETE

## Production Dataset

Target: 100,000
Raw candidates: 100,000
Accepted before deduplication: 100,000
Rejected: 0
Exact duplicates removed: 0
Near duplicates removed: 0
Benchmark matches removed: 0
License rejections: 0
Syntax failures: 0
Execution failures: 0
Final examples: 100,000

## Diversity

Unique families: 100,000
Maximum examples per family: 1
Unique instruction texts: 100,000
Unique response implementations: 6,169
Difficulty: easy 45,009; medium 39,994; hard 14,997
Categories and task types: see `reports/python_100k_categories.json`

## Sources

- `genpy_programmatic_v1`: 100,000 rows; generated provenance; license `generated`; `license_verified=true`.
- Optional CodeSearchNet adapter: not used; unknown/unverified licensing excluded.

## Splits

Train: 90,000
Validation: 5,000
Test: 5,000

Family leakage: 0
Repository leakage: 0

## Validation

Syntax-valid percentage: 100.0%
Execution-tested rows: 100,000
Execution pass percentage: 100.0%
Duplicate IDs: 0
Exact duplicates: 0
Benchmark leakage: 0
Unknown licenses included: 0

## Reproducibility

Seed: 42
Deterministic rebuild: PASS — stable generation order, IDs, splits, and SHA-256 outputs.

## Hashes

all_clean: `52cec86647cb71d6f15ad7c61e357d8084a19733d94ac337587058a9ccedf52d`
train: `17ba25f0154d1ffa04fdd4b91a22123a0770fe6aa76416ba57e4630264cb0b44`
validation: `7ec5fabfb339e0c9986160193da01d30fe6d46035775f81e9777a4ad92731e97`
test: `65716b058821aac3b469827cfd95766c5375c08269d83b65d89ff1cff0c7eb35`

## Tests

Passed: 28
Failed: 0

## Scope Audit

Tokenizer trained: No
Transformer implemented: No
Training started: No
Pretrained weights used: No

## Final Result

Checkpoint 2.5: COMPLETE

Ready for Checkpoint 3: YES
