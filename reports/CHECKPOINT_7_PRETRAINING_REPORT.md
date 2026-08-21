# GenPy Checkpoint 7 Pretraining Report

## Status

PARTIAL — preparation complete; actual Kaggle training intentionally not started.

## Production target

- Model parameters: 201,560,832
- Dataset: 5,398,579 train tokens and 299,166 validation tokens
- Training budget: 1,980 optimizer steps
- Precision: BF16
- Effective tokens/update: 8,192
- Completed optimizer steps: 0
- Tokens processed: 0

## Preflight

The production dry-run passed with the exact model, dataset manifest, 1,980-step
budget, and BF16 configuration. Runtime preflight passed all static checks but
reported `CUDA available: FAIL` and `BF16 supported: FAIL` on this CPU-only
machine. No production optimizer update was executed.

## Measurements

Training and validation losses, throughput, VRAM, resume count, checkpoint
integrity, and final model hash are not applicable until the Kaggle run starts.
They must be populated from `logs/training_metrics.jsonl` and the final
model-only artifact; no values are invented here.

## Verification

- Session-step limit and fixed scheduler budget: PASS
- Checkpoint package/restore with safe extraction: PASS
- Progress inspector: PASS
- Model-only save/reload and weight tying: PASS
- Regression suite: 80 passed, 0 failed

## Warnings

This bounded run is intended to prove learning, stability, validation, and
resume behavior. The 5.4M-token corpus is not sufficient evidence for a fully
pretrained general-purpose 200M-parameter language model, and overfitting must
be reported rather than hidden.

Production training started: No
