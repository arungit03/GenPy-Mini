# GenPy-200M — STEP 8 TODO

States: `[ ]` Not started, `[~]` In progress, `[x]` Completed and verified, `[!]` Blocked.

## Step 8A — Limited pretraining verification

- [x] Record the verified GenPy-200M production model contract.
- [x] Preserve the exact parameter count: `201,560,832`.
- [x] Record the Tesla T4 FP16 hardware and precision configuration.
- [x] Record the real FineWeb-Edu cleaned-corpus subset.
- [x] Record the production 32K tokenizer contract and supplied hash.
- [x] Record packed `uint16` dataset usage and token/document counts.
- [x] Record the 100-step sanity run and validation metrics.
- [x] Record the complete 6,297-step run and final metrics.
- [x] Record the FP16 GradScaler overflow incident and recovery behavior.
- [x] Record safe-checkpoint resume from step 6000 through step 6297.
- [x] Record final checkpoint path and supplied SHA-256 hash.
- [x] Document limitations: approximately 25.8M tokens, sequence length 256, and limited rather than full-scale pretraining.
- [x] Confirm no retraining was performed during documentation cleanup.

## Cleanup and regression verification

- [x] Confirm Step 1–7 functionality remains present.
- [x] Confirm FP16 overflow regression tests remain present.
- [x] Run the complete pytest suite: `102 passed, 1 skipped, 0 failed`.
- [x] Verify the exact production parameter count remains `201,560,832`.
- [x] Confirm architecture, tokenizer, and data pipeline files were not modified.
- [x] Create `docs/STEP8_PRETRAINING.md`.
- [x] Update README Step 8A/8B status.

## Step 8B boundary

- [x] Mark Step 8A complete.
- [x] Leave Step 8B not started.
- [x] Do not implement full-scale pretraining, evaluation/inference, generation, or unrelated Step 8 features.

## Step 8B — Continuation infrastructure prepared

- [x] Add explicit `--init-from-checkpoint` distinct from exact `--resume`.
- [x] Load Step 8A model weights while resetting optimizer, scaler, scheduler, state, sampler, and RNG for a new phase.
- [x] Record source checkpoint provenance in continuation checkpoints.
- [x] Protect fresh Step 8B datasets from Step 8A sampler restoration.
- [x] Start a new warmup/cosine scheduler for Step 8B.
- [x] Add architecture, model-weight, state, sampler, scheduler, and exact-resume regression tests.
- [x] Create `configs/train_step8b_t4.yaml` with conservative continuation values.
- [x] Document the continuation mode and reset policy.
- [x] Confirm Step 8B training was not started.

## Final audit

```text
Total: 32
Completed: 32
Remaining: 0
Blocked: 0
Step 8A: Complete
Step 8B: Not started
```
