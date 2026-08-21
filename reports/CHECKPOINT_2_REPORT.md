# GenPy Checkpoint 2 Report

## Status

PASS — pipeline validation complete; production dataset population pending.

## Pipeline

Schema: PASS

Normalization: PASS

Python syntax validation: PASS

Deduplication: PASS

Near-duplicate detection: PASS

Family leakage prevention: PASS

Deterministic splitting: PASS

Statistics: PASS

## Dataset

Raw examples: 13 fixture records
Accepted: 9 before deduplication
Rejected: 4
Duplicates removed: 2 (1 exact, 1 near)
Near duplicates: 1
Syntax failures: 2 in rejected fixture records
Final examples: 7

Dataset classification: `prototype_only`

Production dataset population: PENDING

## Splits

Train: 5
Validation: 1
Test: 1

Family leakage: 0

## Tests

Passed: 26
Failed: 0

## Output Hashes

Clean dataset: `c93b922ed78e5f81494d53b5b00f85721feff41c680b40772031987d0ebc2cdc`
Train: `7120eb6d1e5d3e3ea596f0cbf5f31fe4b110c9bf7e8223fd36308f106916d78f`
Validation: `befa488a88beae2b8687370d45d7a6bf5facbc3970a9e5e3306fbb28b8809a6c`
Test: `52ca81a517bba41b723c5fecb22285642e9976cbce02d12cd28d8ba8b63a9574`

## Scope Audit

Tokenizer trained: No
Transformer implemented: No
Training started: No
Pretrained weights used: No

## Problems Encountered

No production files exist in `data/raw/`; only the runtime placeholder is present. The smoke fixture intentionally contains invalid Python, obvious Java, an invalid quality score, a missing response, an exact duplicate, and a near duplicate to exercise rejection and audit paths.

## Final Result

Checkpoint 2: COMPLETE
