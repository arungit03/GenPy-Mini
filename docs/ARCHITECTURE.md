# GenPy Architecture Target

This document describes the planned architecture. It is a target for later checkpoints, not an implementation claim.

```text
Tokens
  ↓
Token Embeddings
  ↓
24 Decoder Transformer Blocks
  ↓
Final RMSNorm
  ↓
LM Head
  ↓
Next-token logits
```

Each future decoder block will follow this pre-norm residual structure:

```text
Input
  │
  ├─ RMSNorm
  ├─ Causal Multi-Head Self-Attention + RoPE
  └─ Residual connection
  ├─ RMSNorm
  ├─ SwiGLU MLP
  └─ Residual connection
  ↓
Output
```

## Canonical target values

- 24 Transformer layers
- 768 hidden dimensions
- 12 attention heads
- 64 dimensions per head
- 2176 SwiGLU hidden dimensions
- 32,000-token vocabulary
- 1024-token context
- RoPE positional encoding
- RMSNorm normalization
- Bias-free attention and MLP linear layers
- Tied token embeddings and LM head weights

The Transformer, attention, RoPE mathematics, RMSNorm, SwiGLU, tokenizer, and training components are intentionally deferred to later checkpoints.
