# GenPy-200M Transformer

Checkpoint 4 implements the native PyTorch GenPy decoder-only Transformer. It is randomly initialized and does not load pretrained weights.

## Architecture

```text
Token IDs [B,T]
  -> token embedding [32000, 768]
  -> embedding dropout
  -> 24 x {
       RMSNorm
       causal multi-head self-attention: Q/K/V -> RoPE(Q,K) -> causal SDPA
       residual
       RMSNorm
       SwiGLU: down(SiLU(gate(x)) * up(x))
       residual
     }
  -> final RMSNorm
  -> tied LM head
  -> logits [B,T,32000]
```

The canonical dimensions are 24 layers, hidden size 768, 12 heads, head dimension 64, FFN size 2176, vocabulary 32,000, and maximum context 1,024. The model contains no learned absolute position embeddings. RoPE uses `theta=10000` and cached non-parameter frequency tensors.

## Parameters

| Component | Parameters |
|---|---:|
| Token embedding | 24,576,000 |
| Attention | 56,623,104 |
| SwiGLU | 120,324,096 |
| Block RMSNorm | 36,864 |
| Final RMSNorm | 768 |
| Total unique trainable | 201,560,832 |

The LM head is bias-free and shares the exact `Parameter` object/storage with `token_embedding.weight`; it adds no independent parameters. All Transformer linear layers are bias-free. Parameter-only storage is approximately 768.9 MiB in FP32 or 384.4 MiB in FP16/BF16; these are not training-memory estimates.

## Components and behavior

`RMSNorm` computes the mean-square normalization in FP32 and applies one learned scale vector. `CausalSelfAttention` supports PyTorch SDPA for the model and an eager reference backend for small correctness tests. Attention dropout is active only in training mode; production configuration sets it to zero. `SwiGLU` uses SiLU on the gate branch.

`GenPyForCausalLM(input_ids)` accepts integer IDs shaped `[B,T]`, rejects sequences beyond the configured context and IDs outside the vocabulary, and returns logits shaped `[B,T,vocab_size]`. Generation, KV caching, losses, optimizers, schedulers, packing, and training loops are intentionally outside this checkpoint.

## Initialization and reproducibility

Embeddings and ordinary linear weights use `Normal(0, 0.02)`. Attention output and MLP down projections use `0.02 / sqrt(2 * n_layers)`. RMSNorm scales start at one and all biases are absent. The existing project `set_seed` utility makes tiny-model initialization reproducible under the same PyTorch environment.

## Verification

Run:

```powershell
pytest -q
python scripts/inspect_model.py --config configs/model_200m.yaml
python scripts/verify_model_architecture.py --config configs/model_200m.yaml --tokenizer artifacts/tokenizer/genpy-32k
```

Production training has not started. Deeper numerical, loss, GPU, and overfitting checks belong to Checkpoint 5.
