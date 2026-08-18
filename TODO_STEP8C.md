# GenPy-200M — STEP 8C TODO

States: `[ ]` Not started, `[~]` In progress, `[x]` Completed and verified, `[!]` Blocked.

## Step 8C preparation

- [x] Record the Step 8C T4 continuation configuration.
- [x] Reserve Step 9 evaluation source documents `75000` through `79999`.
- [x] Define the fresh Step 8C source start at `80000`.
- [x] Define the Step 8C target of `250000` source documents.
- [x] Define the intended source boundary as `[80000, 330000)`.
- [x] Document weights-only initialization from the final Step 8B checkpoint.
- [x] Document reset of optimizer, scheduler, scaler, sampler, and RNG state.
- [x] Add configuration, parameter-count, and data-range regression tests.
- [x] Verify existing continuation and checkpoint behavior remains covered.
- [x] Run the complete pytest suite.

## Step 8C execution and verification

- [ ] Prepare the fresh non-overlapping dataset.
- [ ] Validate the dataset.
- [ ] Tokenize the dataset with the unchanged production tokenizer.
- [ ] Record the exact train token count.
- [ ] Calculate `max_steps = floor(train_token_count / 4096)`.
- [ ] Run the GPU dry run.
- [ ] Run the 100-step Tesla T4 sanity run.
- [ ] Verify validation behavior and metrics.
- [ ] Run full Step 8C training.
- [ ] Verify exact resume behavior.
- [ ] Record the final checkpoint SHA-256.
- [ ] Persist the final Kaggle artifact.
- [ ] Run the post-Step 8C Step 9 benchmark comparison.

## Scope boundary

- [x] Do not start Step 8C training during preparation.
- [x] Do not download the 250000-document dataset locally during preparation.
- [x] Preserve the exact `201,560,832` production parameter count.
- [x] Preserve model architecture, tokenizer, checkpoint format, and Step 9 evaluation mathematics.

## Final audit

```text
Preparation: Complete
Training: Not started
Remaining execution items: 13
Blocked items: 0
Protected Step 9 range: [75000, 80000)
Step 8C range: [80000, 330000)
```
