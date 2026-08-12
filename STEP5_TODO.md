# GenPy-200M â€” STEP 5 TODO

States: `[ ]` Not started, `[~]` In progress, `[x]` Completed and verified, `[!]` Blocked.

## TODO 1 â€” Repository Audit
- [x] Inspected README, STEP1â€“STEP4 TODO files, architecture docs, model config, model modules, tests, and `GENPY_STEP5_GPU_REPORT.txt`.
- [x] Baseline regression: `64 passed`, `0 failed`.
- [x] Confirmed the locked production architecture and recorded GPU evidence are present.

## TODO 2 â€” Create STEP5_TODO.md
- [x] Created this persistent checklist without modifying historical TODO files.

## TODO 3 â€” Verification Package
- [x] Create small reusable `genpy/verification/` modules for loss, finiteness, and diagnostics.

## TODO 4 â€” Causal LM Loss Helper
- [x] Implement shifted causal cross-entropy for logits `[B,T,V]` and labels `[B,T]`.
- [x] Validate incompatible shapes without adding optimizer or training-loop logic.

## TODO 5 â€” Loss Reference Verification
- [x] Compare helper output against `log_softmax` plus gathered target log-probabilities.
- [x] Test the causal shift explicitly.

## TODO 6 â€” Finite Tensor Utilities
- [x] Report NaN, Inf, finite state, parameter names, missing gradients, and non-finite gradients.
- [x] Do not silently discard failures.

## TODO 7 â€” Gradient Health Verification
- [x] Run tiny forward, causal loss, backward, and verify every trainable parameter has finite gradients.

## TODO 8 â€” Causal Isolation Verification
- [x] Verify future-token changes do not affect earlier outputs and changed positions normally differ.

## TODO 9 â€” RoPE Deep Verification
- [x] Verify shapes, finiteness, position zero, later rotation, gradients, no trainable parameters, odd dimensions, context limits, and production length 1025 rejection.

## TODO 10 â€” RMSNorm Stability Verification
- [x] Verify normal, 1e-5-scale, and 1e3-scale inputs have finite outputs and gradients.
- [x] Run BF16 checks only when supported; CPU pytest must remain valid.

## TODO 11 â€” Forward Correctness Verification
- [x] Verify tiny `[2,16]` input produces finite `[2,16,256]` logits.

## TODO 12 â€” Production Construction Verification
- [x] Verify production construction, exact parameter count, and all locked dimensions without a long CPU forward.

## TODO 13 â€” Weight Tying Verification
- [x] Verify object identity, storage identity, mutation propagation, restoration, and state-dict tying.

## TODO 14 â€” Optimization-Like Weight-Tying Test
- [x] Perform one tiny SGD-like update and verify tying remains intact without creating a trainer.

## TODO 15 â€” Tiny-Batch Overfit Verification Script
- [x] Create a tiny-only verification script with a fixed batch and temporary AdamW loop.
- [x] Require substantial loss reduction without encoding exact GPU values.

## TODO 16 â€” GPU Smoke Test Script
- [x] Create optional CUDA script with safe batch/sequence defaults, fp32/BF16 support, optional backward, and graceful CPU behavior.

## TODO 17 â€” Production Context Limit Verification
- [x] Verify context 1024 is accepted and 1025 rejected without requiring a full CPU backward.

## TODO 18 â€” Diagnostics
- [x] Report parameters, trainable count, finite state, gradients, tying, and model configuration.

## TODO 19 â€” verify_model.py
- [x] Create CPU-safe verification CLI covering production contract, tiny forward/loss/backward, gradients, causality, and finiteness.
- [x] Return non-zero on mandatory failures.

## TODO 20 â€” Memory Documentation
- [x] Document weights-only FP32/FP16/BF16 estimates and measured Tesla T4 observations from the supplied report.

## TODO 21 â€” Preserve External GPU Evidence
- [x] Confirmed `GENPY_STEP5_GPU_REPORT.txt` exists and will be preserved as recorded evidence.
- [x] Copy verified information into verification documentation without inventing local GPU results.

## TODO 22 â€” Numerical Stability Test
- [x] Add RMSNorm, RoPE, tiny logits, loss, gradient, and graceful BF16 stability tests.

## TODO 23 â€” test_model_verification.py
- [x] Add an integrated tiny-model contract test covering forward, loss, backward, finite values, gradients, tying, and causality.

## TODO 24 â€” No Production Training
- [x] Audit that Step 5 adds no production tokenizer, packing, dataloader, trainer, optimizer builder, scheduler, checkpoint manager, distributed training, pretraining, generation, or sampling.

## TODO 25 â€” No Architecture Drift
- [x] Confirm all locked production dimensions, parameter count, RMSNorm, RoPE, causal MHA, bias-free projections, and tying remain unchanged.

## TODO 26 â€” Documentation
- [x] Create `docs/VERIFICATION.md` documenting CPU checks, numerical checks, loss, gradients, causality, tying, overfit, GPU evidence, memory, limitations, and the Step 6 boundary.

## TODO 27 â€” README
- [x] Mark Step 5 in progress initially and add a concise verification summary.
- [x] Mark Step 5 complete only after all permanent checks pass.

## TODO 28 â€” Focused Tests
- [x] Run all five required Step 5 focused test files individually.

## TODO 29 â€” Existing Regression Suite
- [x] Run the complete suite with zero failures and preserve all historical tests.

## TODO 30 â€” verify_model.py Execution
- [x] Run the verification CLI and record its actual CPU-safe result.

## TODO 31 â€” Tiny Overfit Script Execution
- [x] Run the tiny overfit script and record actual local initial/final loss and reduction.

## TODO 32 â€” GPU Script Static/Graceful Validation
- [x] Run the GPU script locally and report graceful CUDA-unavailable behavior if applicable.

## TODO 33 â€” Final Architecture Audit
- [x] Run parameter counting and recheck tying and bias-free projections.

## TODO 34 â€” Final Regression
- [x] Run final pytest and record the exact pass/fail result.

## TODO 35 â€” Final TODO Audit
- [x] Read this checklist completely and report Total, Completed, Remaining, and Blocked.

## TODO 36 â€” Mark Step 5 Complete
- [x] Only after every mandatory check passes, mark Step 5 complete and leave Step 6 not started.

## Scope rules

- Use PyTorch primitives and keep verification separate from the future training engine.
- CPU pytest must not require CUDA; GPU checks must detect capabilities and skip gracefully.
- Preserve the supplied Kaggle evidence and never invent GPU results.
- Do not change the locked architecture unless a verified correctness bug requires it.
- Do not start Step 6.

## Final audit

All 36 mandatory Step 5 TODOs are completed and verified.

```text
Baseline regression: 64 passed, 0 failed
Final regression: 76 passed, 1 skipped, 0 failed
Tiny overfit: 5.538577 -> 0.005208, 99.9% reduction, PASS
Local CUDA smoke: skipped gracefully; CUDA unavailable
Total: 36
Completed: 36
Remaining: 0
Blocked: 0
```

