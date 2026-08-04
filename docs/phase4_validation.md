# Phase 4 Validation

Results below are actual Windows CPU results from 2026-08-03. They validate infrastructure, not
model quality or Phase 5 readiness.

## Readiness

The GenPy-5M gate returns `READY_FOR_SMOKE_MODEL` with exit status 1 because the frozen
16,384-token production tokenizer, matching exact counts, and production packs do not exist. The
smoke model gate returns `READY_FOR_SMOKE_MODEL` with exit status 0.

## Model

The smoke model has 147,776 unique tied parameters. A float32 forward produced logits
`[1, 64, 1024]`, loss 6.8912339 over 63 active targets, finite gradients, and an optimizer
parameter change. Strict causality, length-one behavior, maximum context, invalid IDs, aligned
loss, RMSNorm, RoPE, SwiGLU, and optimized/reference attention passed tests.

The bounded safe-fixture micro-overfit ran 20 steps. Loss changed from 6.8912339 to 2.1843474,
a 68.3025% reduction, without NaNs. This is correctness evidence only.

## Packing

Seven original safe fixture records produced seven smoke samples across six independent shards.
Binary payload was 1,358 bytes. The stream contained 317 unique serialized tokens, 137 padding
positions, 261 active targets, 187 masked targets, and one masked cross-record transition. Overall
unique-token packing efficiency was 69.67%; padding was 30.11%. Repeated binary output matched.

| Group | Records | Samples | Real tokens | Padding | Active targets |
| --- | ---: | ---: | ---: | ---: | ---: |
| pretraining train | 2 | 2 | 101 | 28 | 99 |
| pretraining validation | 1 | 1 | 29 | 36 | 28 |
| pretraining test | 1 | 1 | 29 | 36 | 28 |
| instruction train | 1 | 1 | 48 | 17 | 29 |
| instruction validation | 1 | 1 | 47 | 18 | 32 |
| instruction test | 1 | 1 | 63 | 2 | 45 |

All six checksums, shapes, ranges, masks, fingerprints, config hashes, and split references
validated. Deliberate corruption was rejected. No Phase 2 production data was packed.

## Memory Estimates

At sequence 1,024 and micro-batch one, estimated totals with AdamW are 129,018,880 bytes for
GenPy-5M float32, 528,603,136 for GenPy-25M float32, and 2,164,584,448 for GenPy-100M float32.
The 100M float16/bfloat16 estimate is 1,862,594,560 bytes. Activation memory is analytical and
these values do not guarantee GPU fit.
