# GenPy-200M — STEP 9 TODO

States: `[ ]` Not started, `[~]` In progress, `[x]` Completed and verified, `[!]` Blocked.

## Step 9.7 — Production inference and evaluation tooling

- [x] Add reusable production generation utilities under `genpy/inference/`.
- [x] Support greedy decoding.
- [x] Support temperature sampling.
- [x] Support top-k filtering.
- [x] Support top-p / nucleus filtering.
- [x] Support repetition penalties.
- [x] Support configurable maximum new tokens.
- [x] Support EOS stopping.
- [x] Support configurable forbidden PAD/BOS/UNK generation.
- [x] Truncate generation context to `model.config.max_seq_len`.
- [x] Use `torch.inference_mode()` for generation.
- [x] Support seeded deterministic generation.
- [x] Support CPU and CUDA device selection.
- [x] Support FP32, FP16, and BF16 where the device supports them.
- [x] Add strict model-only checkpoint loading for inference.
- [x] Add `scripts/generate.py`.
- [x] Add reusable packed-dataset evaluation utilities under `genpy/evaluation/`.
- [x] Compute exact token-weighted FP32 negative log likelihood.
- [x] Compute perplexity as `exp(loss)`.
- [x] Report evaluated tokens and complete evaluation windows.
- [x] Add `scripts/evaluate.py` with JSON output support.
- [x] Record the verified unseen FineWeb-Edu evaluation baseline.
- [x] Add CPU-friendly regression tests for inference and evaluation.
- [x] Run the complete pytest suite with zero failures.
- [x] Verify the production parameter count remains `201,560,832`.
- [x] Preserve the tokenizer, model architecture, training engine, and checkpoint format.

## Scope boundary

- [x] Do not retrain the model.
- [x] Do not begin Step 8C or further pretraining.
- [x] Keep the real 2.3GB checkpoint and Kaggle data out of CPU tests.
- [ ] Complete any later Step 9 release or serving work not covered by Step 9.7.

## Final audit

```text
Step 9.7: Complete
Remaining Step 9.7 items: 0
Blocked Step 9.7 items: 0
Later Step 9 work: Not started
```
