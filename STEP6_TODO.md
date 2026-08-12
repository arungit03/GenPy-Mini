# GenPy-200M â€” STEP 6 TODO

States: `[ ]` Not started, `[~]` In progress, `[x]` Completed and verified, `[!]` Blocked.

## Repository and configuration

- [x] TODO 1 â€” Create this checklist and preserve historical TODO files.
- [x] TODO 2 â€” Audit the existing training configuration and record the baseline.
- [x] TODO 3 â€” Extend and validate TrainingConfig with engine settings.

## Training data

- [x] TODO 4 â€” Use validated uint16 token storage.
- [x] TODO 5 â€” Implement streaming tokenized-data preparation.
- [x] TODO 6 â€” Preserve document boundaries with EOS.
- [x] TODO 7 â€” Support streaming CLI limits and split processing.
- [x] TODO 8 â€” Write token metadata and provenance.
- [x] TODO 9 â€” Create token files atomically.
- [x] TODO 10 â€” Implement memory-mapped packed-token dataset.
- [x] TODO 11 â€” Implement contiguous sequence-plus-target packing rules.
- [x] TODO 12 â€” Test dataset length, dtype, windows, EOS, tails, mmap, metadata, and IDs.
- [x] TODO 13 â€” Implement deterministic stateful sample ordering.
- [x] TODO 14 â€” Implement deterministic DataLoader helpers.
- [x] TODO 55 â€” Verify temporary training-data preparation end to end.

## Optimization and scheduling

- [x] TODO 15 â€” Reuse verified next-token loss semantics for packed targets.
- [x] TODO 16 â€” Create AdamW from TrainingConfig.
- [x] TODO 17 â€” Create unique weight-decay parameter groups.
- [x] TODO 18 â€” Test optimizer coverage and settings.
- [x] TODO 19 â€” Implement optimizer-step warmup/cosine scheduling.
- [x] TODO 20 â€” Implement a pure testable LR function, including zero warmup.
- [x] TODO 21 â€” Test scheduler boundaries and serialization.

## Precision and state

- [x] TODO 22 â€” Implement FP32/FP16/BF16/auto precision management.
- [x] TODO 23 â€” Reject unsupported explicit precision safely.
- [x] TODO 24 â€” Implement serializable TrainingState.
- [x] TODO 25 â€” Implement correct gradient accumulation.
- [x] TODO 26 â€” Test accumulation equivalence and update counts.
- [x] TODO 27 â€” Implement post-unscale gradient clipping.
- [x] TODO 28 â€” Test clipping and finite gradient norms.

## Metrics, logging, validation

- [x] TODO 29 â€” Implement training metrics and perplexity safety.
- [x] TODO 30 â€” Implement console and JSONL logging.
- [x] TODO 31 â€” Implement limited no-grad validation with mode restoration.
- [x] TODO 32 â€” Test validation averaging and immutability.

## Checkpoints and engine

- [x] TODO 33 â€” Implement complete checkpoint state.
- [x] TODO 34 â€” Save checkpoints atomically.
- [x] TODO 35 â€” Track a deterministic latest checkpoint pointer.
- [x] TODO 36 â€” Retain the configured number of checkpoints safely.
- [x] TODO 37 â€” Validate compatibility on checkpoint load.
- [x] TODO 38 â€” Save and restore Python, NumPy, Torch, and CUDA RNG states.
- [x] TODO 39 â€” Verify deterministic continuous/resumed training.
- [x] TODO 40 â€” Implement the modular TrainingEngine.
- [x] TODO 41 â€” Define an explicit engine API.
- [x] TODO 42 â€” Verify training-step order.
- [x] TODO 43 â€” Stop safely on non-finite loss or gradients.
- [x] TODO 44 â€” Implement the explicit-max-steps training CLI.
- [x] TODO 45 â€” Validate and print CLI startup safety information.
- [x] TODO 46 â€” Report effective sequences and tokens per update.
- [x] TODO 47 â€” Implement the tiny end-to-end training smoke script.
- [x] TODO 48 â€” Demonstrate smoke checkpoint resume.
- [x] TODO 49 â€” Construct the production engine without serious training.

## Documentation, audits, and final verification

- [x] TODO 50 â€” Confirm no tokenizer or architecture drift.
- [x] TODO 51 â€” Create docs/TRAINING_ENGINE.md.
- [x] TODO 52 â€” Update README progress and summary.
- [x] TODO 53 â€” Run every focused Step 6 test individually.
- [x] TODO 54 â€” Run the complete historical regression suite.
- [x] TODO 56 â€” Run the smoke script and record actual results.
- [x] TODO 57 â€” Implement and test train CLI dry-run validation.
- [x] TODO 58 â€” Verify CPU-only operation.
- [x] TODO 59 â€” Keep CUDA optional and Step 7-bound.
- [x] TODO 60 â€” Audit trusted-path checkpoint loading and robustness.
- [x] TODO 61 â€” Preserve generated-artifact gitignore rules.
- [x] TODO 62 â€” Verify the exact parameter count.
- [x] TODO 63 â€” Run the Step 5 verification regression.
- [x] TODO 64 â€” Complete the training-engine audit.
- [x] TODO 65 â€” Complete the scope audit: no pretraining, distributed training, generation, or tokenizer retraining.
- [x] TODO 66 â€” Run final pytest and record exact counts.
- [x] TODO 67 â€” Read this checklist and reach zero remaining/blocked items.
- [x] TODO 68 â€” Mark Step 6 complete and leave Step 7 not started.

## Scope boundary

Step 6 builds the reusable engine only. It must not start meaningful
GenPy-200M pretraining, Step 7 experiments, distributed training, generation,
sampling, instruction tuning, or tokenizer retraining.

## Final audit

```text
Baseline regression: 76 passed, 1 skipped, 0 failed
Final regression: 94 passed, 1 skipped, 0 failed
Production parameter count: 201,560,832, difference 0
Training smoke: 4 steps, 4.145567 -> 4.004988, 128 tokens, PASS
Deterministic resume: PASS
Local CUDA smoke: skipped gracefully; CUDA unavailable
Total: 68
Completed: 68
Remaining: 0
Blocked: 0
```

