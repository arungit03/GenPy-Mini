# ADR-005: Decoder-Only Transformer

Status: accepted for Phase 4 infrastructure; architecture freeze pending production tokenizer.

GenPy uses a bias-free, pre-normalized decoder-only Transformer with RMSNorm, standard multi-head
causal self-attention, RoPE, SwiGLU, tied token/output weights, and random initialization. It has
no encoder, cross-attention, learned positions, grouped-query attention, mixture of experts,
classification head, or pretrained component.

The complete Phase 1 profiles take precedence over the prompt's fallback profiles. Therefore the
established 4/192/4, 8/384/6, and 12/768/12 configurations remain unchanged. Exact counts are
4,916,928, 20,453,760, and 97,536,768. Residual dropout remains the Phase 1 value 0.1; other
dropouts default to zero.

All model checkpoints must include and validate the model config hash, tokenizer fingerprint,
vocabulary, special IDs, context, parameter count, and tensor checksum. Architecture freeze
requires the missing production tokenizer fingerprint.
