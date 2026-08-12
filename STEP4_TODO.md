# GenPy-200M â€” STEP 4 TODO

States: `[ ]` Not started, `[~]` In progress, `[x]` Completed and verified, `[!]` Blocked.

## TODO 1 â€” Audit Existing Repository
- [x] Steps 1â€“3 files, README, docs, configuration, and model scaffolding inspected.
- [x] Existing tests pass: baseline `45 passed`.
- [x] `configs/model_200m.yaml` exists with vocabulary `32000` and the locked architecture values.
- [x] Tokenizer contract remains vocabulary `32000` with PAD/BOS/EOS/UNK IDs `0/1/2/3`.
- [x] No model implementation exists beyond scaffolding and no Step 4 training engine exists.

## TODO 2 â€” Create STEP4_TODO.md
- [x] This persistent checklist was created without changing historical TODO files.

## TODO 3 â€” Create Model Module Structure
- [x] Add `rmsnorm.py`, `rope.py`, `attention.py`, `swiglu.py`, `block.py`, `model.py`, and `initialization.py` under `genpy/model/`.
- [x] Add focused component, architecture, tying, and parameter-count tests.
- [x] Do not add training modules.

## TODO 4 â€” Extend Model Configuration
- [x] Add the locked dropout, bias, and initializer fields while preserving all existing locked values.

## TODO 5 â€” Extend ModelConfig Validation
- [x] Validate positive dimensions/scalars, dropout ranges, initializer range, and hidden/head relationships.

## TODO 6 â€” Implement RMSNorm
- [x] Implement bias-free PyTorch RMSNorm with mixed-precision-safe normalization.

## TODO 7 â€” Test RMSNorm
- [x] Verify shape, dtype, finite values, gradients, scale shape, no bias, and numerical behavior on a tiny size.

## TODO 8 â€” Implement Rotary Embedding Frequency Cache
- [x] Implement non-trainable inverse-frequency and cosine/sine caches for the configured context.

## TODO 9 â€” Implement RoPE Application
- [x] Apply RoPE only to Q/K, preserve shapes, support the attention layout, and reject lengths beyond context.

## TODO 10 â€” Test RoPE
- [x] Verify shape, Q/K changes, no V involvement, position zero, determinism, gradients, odd-dimension rejection, context rejection, and no trainable parameters.

## TODO 11 â€” Implement Causal Self-Attention Projections
- [x] Implement standard bias-free Q/K/V/O projections; no GQA, MQA, MLA, MoE, or cross-attention.

## TODO 12 â€” Implement Head Reshaping
- [x] Reshape `[B,T,hidden]` to `[B,heads,T,head_dim]` and merge heads back without unnecessary copies.

## TODO 13 â€” Apply RoPE Inside Attention
- [x] Apply RoPE to Q/K before score computation.

## TODO 14 â€” Implement Causal Attention
- [x] Implement scaled dot-product attention, causal masking, softmax, value aggregation, head merge, and O projection.

## TODO 15 â€” Attention Mask Efficiency
- [x] Use a reusable triangular causal mask and keep the implementation ready for a later optimized backend.

## TODO 16 â€” Implement Attention Dropout Interface
- [x] Honor configured attention dropout through normal PyTorch train/eval behavior, defaulting to zero.

## TODO 17 â€” Test Causal Attention
- [x] Verify shape, gradients, finiteness, bias-free projections, dimensions, RoPE, and mandatory causality behavior.

## TODO 18 â€” Implement SwiGLU
- [x] Implement bias-free gate/up/down projections with `SiLU(gate) * up` and the configured intermediate size.

## TODO 19 â€” Test SwiGLU
- [x] Verify shape, gradients, dimensions, no bias, formula, and deterministic eval behavior.

## TODO 20 â€” Implement GenPy Transformer Block
- [x] Implement the pre-norm attention and SwiGLU residual structure with two independent RMSNorm modules and residual dropout.

## TODO 21 â€” Test Transformer Block
- [x] Verify shape, independent norms, attention, SwiGLU, residual paths, gradients, and finite values.

## TODO 22 â€” Implement Token Embedding
- [x] Add the configured token embedding without learned positional embeddings or embedding scaling.

## TODO 23 â€” Implement Embedding Dropout Interface
- [x] Honor configured embedding dropout through normal PyTorch behavior, defaulting to zero.

## TODO 24 â€” Implement Full GenPy Model
- [x] Implement embedding â†’ blocks â†’ final RMSNorm â†’ LM head and logits only, with no loss calculation.

## TODO 25 â€” Input Validation
- [x] Validate rank-2 token IDs, context length, and clear invalid token IDs.

## TODO 26 â€” Tie LM Head to Token Embedding
- [x] Use a bias-free LM head whose weight is the same parameter as the token embedding.

## TODO 27 â€” Test Weight Tying
- [x] Verify object/storage identity, mutation visibility, and one shared optimizer parameter.

## TODO 28 â€” Implement Model Initialization
- [x] Initialize embeddings/linear weights from the configured normal range and RMSNorm scales to one.

## TODO 29 â€” Residual Projection Scaling
- [x] Initialize attention O and SwiGLU down projections with `initializer_range / sqrt(2 * num_layers)` and document it.

## TODO 30 â€” Test Initialization
- [x] Verify scale/means/stds, residual scaling, and absence of NaN/Inf on a small model.

## TODO 31 â€” Expose Clean Model API
- [x] Export the public model components from `genpy.model`.

## TODO 32 â€” Extend Parameter Count Script
- [x] Report theoretical, actual trainable, actual total, difference, breakdown, and weights-only memory estimates.

## TODO 33 â€” Add Parameter Breakdown
- [x] Report embedding, attention, SwiGLU, RMSNorm, and other counts without double-counting tied weights.

## TODO 34 â€” Verify Exact Production Parameter Count
- [x] Instantiate the production model and verify actual trainable count is `201,560,832`.

## TODO 35 â€” Check Number of Blocks
- [x] Programmatically verify exactly 24 production blocks.

## TODO 36 â€” Verify Attention Dimensions
- [x] Verify every production attention layer has 12 heads, head dimension 64, and hidden size 768.

## TODO 37 â€” Verify SwiGLU Dimensions
- [x] Verify every production gate/up/down projection uses 768â†”2176 as specified.

## TODO 38 â€” Verify Bias-Free Architecture
- [x] Verify every specified production linear module has `bias is None`.

## TODO 39 â€” Verify No Learned Positional Embeddings
- [x] Verify the production model contains no learned positional embedding table.

## TODO 40 â€” Tiny Configuration Fixture
- [x] Add and use a tiny configuration for unit tests while retaining the same architecture concepts.

## TODO 41 â€” Tiny Full-Model Forward Test
- [x] Verify tiny `[2,16]` input produces finite `[2,16,vocab]` logits.

## TODO 42 â€” Production Model Construction Test
- [x] Verify full CPU construction, counting, and architecture inspection without training.

## TODO 43 â€” Short Production Forward Smoke Test
- [x] If memory permits, verify a no-grad batch-1 short-context production forward; report honestly if skipped.

## TODO 44 â€” Model Memory Estimate
- [x] Report weights-only FP32 and FP16/BF16 estimates and distinguish them from training memory.

## TODO 45 â€” Architecture Documentation
- [x] Create `docs/ARCHITECTURE.md` covering model, blocks, attention, RoPE, SwiGLU, RMSNorm, tying, initialization, and Step 4 boundaries.

## TODO 46 â€” Update README Architecture Section
- [x] Document the implemented architecture and verified parameter count; leave Step 4 in progress until final validation.

## TODO 47 â€” Model Representation
- [x] Provide a useful concise model representation/summary.

## TODO 48 â€” No External Transformer Implementations
- [x] Audit and verify no forbidden transformer implementation or attention dependency is used.

## TODO 49 â€” No Pretrained Weights
- [x] Audit and verify no pretrained weights or downloads are used.

## TODO 50 â€” Architecture Scope Audit
- [x] Verify Step 4 contains no optimizer, scheduler, training loop, mixed precision training, checkpoint manager, tokenization, packing, loss loop, sampling, or generation.

## TODO 51 â€” Run Component Tests
- [x] Run focused RMSNorm, RoPE, attention, SwiGLU, and block tests.

## TODO 52 â€” Run Model Tests
- [x] Run architecture, weight-tying, and parameter-count tests.

## TODO 53 â€” Run Complete Regression Suite
- [x] Run all tests with zero failures and preserve Steps 1â€“3.

## TODO 54 â€” Run Parameter Count Script
- [x] Run `python scripts/count_parameters.py` and record the actual result.

## TODO 55 â€” Environment Regression
- [x] Run `python scripts/check_environment.py`; CPU-only development remains valid.

## TODO 56 â€” Fresh Process Import Test
- [x] Verify clean-process imports, tiny construction, and tiny forward.

## TODO 57 â€” Check State Dict Weight Tying
- [x] Verify save/load preserves tied-weight architecture.

## TODO 58 â€” Serialization Smoke Test
- [x] Verify tiny state-dict round trip produces identical eval logits without committed artifacts.

## TODO 59 â€” Deterministic Eval Test
- [x] Verify repeated eval forwards with fixed inputs are identical.

## TODO 60 â€” Final Architecture Audit
- [x] Programmatically report and verify all locked production values, tying, bias-free projections, and count.

## TODO 61 â€” Final Full Tests
- [x] Run final pytest, parameter count script, and environment checker with no regressions.

## TODO 62 â€” Update README Completion
- [x] Only after every mandatory validation succeeds, mark Step 4 complete and Step 5 not started.

## TODO 63 â€” Final TODO Audit
- [x] Read this checklist end-to-end and reach Remaining `0`, Blocked `0`.

## Scope and quality rules

- Use only PyTorch primitives and beginner-readable modular code.
- Keep all specified projections bias-free, use pre-norm, tie embeddings, and use RoPE without learned positional embeddings.
- Do not add optimizer, scheduler, training loop, pretraining, generation, sampling, or pretrained weights.
- Investigate every failure, run focused verification, then broader regression tests before marking work complete.

## Final audit

All 63 mandatory TODOs are completed and verified.

```text
Total: 63
Completed: 63
Remaining: 0
Blocked: 0
```

