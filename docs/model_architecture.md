# GenPy Model Architecture

GenPy is a decoder-only autoregressive Transformer implemented directly in PyTorch. It predicts
the next token from earlier tokens under a causal mask. No pretrained weights, embeddings,
layers, adapters, model code, or tokenizers are loaded.

Each token ID indexes a trainable embedding. The residual stream passes through pre-normalized
blocks:

```text
x = x + attention(rms_norm_1(x))
x = x + swiglu(rms_norm_2(x))
```

RMSNorm scales by root mean square without subtracting the mean. Multi-head self-attention uses
RoPE on queries and keys, PyTorch scaled-dot-product attention when available, and a tested
reference path. Causality prevents future-token access. There are no learned position
embeddings, encoder, cross-attention, classification head, grouped-query attention, or KV cache.

SwiGLU computes `down(silu(gate(x)) * up(x))`. Linear layers have no bias. Token embeddings and
the language-model output projection share the same parameter object. All weights begin from a
seeded random normal distribution; RMSNorm scales begin at one.

Production context length is exactly 1,024. The smoke model uses 64 solely for CPU correctness.
The established Phase 1 profiles remain the source of truth:

| Model | Layers | Width | Heads | Head dim | FFN | Exact parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GenPy-5M | 4 | 192 | 4 | 48 | 512 | 4,916,928 |
| GenPy-25M | 8 | 384 | 6 | 64 | 1,024 | 20,453,760 |
| GenPy-100M | 12 | 768 | 12 | 64 | 2,048 | 97,536,768 |

The tokenizer fingerprint is compatibility metadata, not a numerical model input. A checkpoint
must match model config hash, tokenizer identity, vocabulary, special IDs, context length,
parameter count, tensor shapes, and weight checksum before loading.

Known limitations include no inference cache, no generation API, no distributed training engine,
and no production training. Analytical memory estimates do not guarantee hardware fit.
