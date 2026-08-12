# GenPy-200M Step 5 Verification

Step 5 verifies the Step 4 decoder-only architecture without implementing the
Step 6 training engine. Verification is CPU-safe and uses only small temporary
optimization logic in the tiny overfit script.

## CPU verification

`python scripts/verify_model.py` verifies the production parameter count,
24-block architecture contract, dimensions, bias-free projections, tied
weights, tiny forward pass, causal loss, backward pass, finite gradients, and
causal isolation. The local CPU run passed.

The tiny model uses vocabulary 256, context 32, hidden size 64, two layers,
four heads, head dimension 16, and SwiGLU intermediate size 128.

## Loss correctness

`genpy.verification.loss.causal_lm_loss` shifts logits and labels so position
`t` predicts token `t+1`. Tests compare PyTorch cross-entropy with a manual
log-softmax/gather reference and explicitly test the shift.

## Gradient verification

The tiny forward/loss/backward check requires every trainable parameter to have
a gradient and requires every gradient and loss value to be finite.

## Causal isolation

Changing a future token must not change earlier logits. The tests use strict
`1e-6` tolerances and also verify that the changed position responds.

## RoPE and RMSNorm

RoPE tests verify shape preservation, finite values, position-zero identity,
later-position rotation, gradients, no trainable parameters, even head
dimensions, and context rejection. RMSNorm tests cover normal values,
approximately `1e-5` values, approximately `1e3` values, finite outputs, and
finite input/weight gradients. CUDA BF16 tests skip when unavailable.

## Weight tying

The embedding and LM head remain the same parameter and storage after a
temporary update, one tiny SGD-like update, and state-dict serialization.

## Tiny-batch overfit

The local fixed-batch verification script used 200 steps and reported:

```text
Initial loss: 5.538577
Final loss: 0.005208
Reduction: 99.9%
Result: PASS
```

This is a verification experiment, not a reusable trainer.

## Memory and GPU evidence

Theoretical raw parameter memory for 201,560,832 parameters is approximately:

```text
FP32 weights: 768.89 MiB (about 769 MiB)
FP16/BF16 weights: 384.45 MiB (about 384 MiB)
```

These are **weights only**. They exclude gradients, optimizer state, and
activations.

The repository preserves `GENPY_STEP5_GPU_REPORT.txt` as recorded external
evidence from a Tesla T4, batch size 1, and the current implementation:

```text
FP32 sequence 32: forward/backward PASS, peak ~1.772 GiB
BF16 sequence 32: forward/backward PASS, peak ~1.765 GiB
Sequence 512: peak ~2.363 GiB
Sequence 1024: peak ~4.333 GiB
Full [1,1024] BF16 forward/backward: PASS
Missing gradients: 0
NaN/Inf gradients: 0
```

These measurements are specific to that Tesla T4 run and are not universal
hardware guarantees. The local machine has no CUDA, so no local GPU result is
claimed.

## Limitations and Step 6 boundary

Step 5 does not add a production dataloader, optimizer builder, scheduler,
gradient accumulation, checkpoint manager, distributed training, pretraining,
generation, or sampling. Step 6 will implement the GenPy training engine.
