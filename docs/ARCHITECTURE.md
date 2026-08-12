# GenPy-200M Architecture

GenPy-200M is a decoder-only Transformer implemented from PyTorch primitives.
Step 4 contains the model architecture and its tests only; it does not contain
training, generation, or pretrained weights.

## Model

| Property | Value |
| --- | ---: |
| Vocabulary | 32,000 |
| Maximum context | 1,024 |
| Hidden size | 768 |
| Transformer blocks | 24 |
| Attention heads | 12 |
| Head dimension | 64 |
| SwiGLU intermediate size | 2,176 |
| RoPE theta | 10,000 |
| Normalization epsilon | 1e-5 |
| Linear biases | None |
| Embedding/LM-head tying | Enabled |

The verified trainable parameter count is **201,560,832**. The tied LM head
does not create a second vocabulary-by-hidden-size parameter matrix.

## Transformer block

Each block is pre-normalized:

```text
x = x + attention(attn_norm(x))
x = x + swiglu(ffn_norm(x))
```

The attention and feed-forward residual branches use configurable dropout,
which is zero in the locked GenPy-200M configuration.

## Attention

GenPy uses standard multi-head self-attention. Four bias-free linear
projections create Q, K, V, and O. Q and K are reshaped to 12 heads of 64
values each, rotary position embeddings are applied to Q and K, and a reusable
lower-triangular mask prevents each token from attending to future tokens.

## RoPE

Rotary position embeddings use theta 10,000 and a non-trainable frequency cache
for the 1,024-token context. No learned positional embedding table is present.

## SwiGLU

The feed-forward network uses three bias-free projections:

```text
down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

The projections are 768 → 2,176, 768 → 2,176, and 2,176 → 768.

## Normalization

RMSNorm computes:

```text
RMS(x) = sqrt(mean(x²) + eps)
output = weight * x / RMS(x)
```

It has one trainable scale vector and no bias. Half-precision inputs are
normalized in a safer computation dtype and returned in their original dtype.

## Weight tying

`token_embedding.weight` and `lm_head.weight` are the same PyTorch parameter.
This preserves shared storage, avoids double-counting, and leaves one shared
parameter for a later optimizer.

## Initialization

Embedding and linear weights use a normal distribution with standard deviation
0.02 by default. RMSNorm scales start at one. Attention O projections and
SwiGLU down projections use the documented depth-aware standard deviation
`0.02 / sqrt(2 * 24)` to temper residual updates.

## Design boundaries

Step 4 intentionally does not implement an optimizer, scheduler, training
loop, mixed-precision training, checkpoint manager, tokenization, sequence
packing, loss computation, sampling, or text generation. Those belong to later
steps.
