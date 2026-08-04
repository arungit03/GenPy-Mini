# ADR-006: Token Packing And Loss Masks

Status: accepted for Phase 4 infrastructure.

Production stores 1,025 `uint16` tokens and 1,024 `uint8` loss flags per sample. Input and labels
are adjacent slices with stride 1,024. Smoke uses the same `context+1` rule at 65/64. Families and
splits have independent shards and manifests.

Records may share causal context. EOS/BOS remain visible, but a next-record BOS target is masked.
No block-diagonal document isolation is implemented. Final partial blocks are padded and all
padding targets are masked.

Pretraining defaults to full-LM loss. Instruction defaults to assistant-only loss beginning at
the `<|assistant|>` target and including the code boundary, answer, `<|end|>`, and `<|eos|>`.
Prompts remain visible but inactive. Policies never change silently.

Every shard stores the tokenizer fingerprint and packing config hash. A mismatch, cross-split
source reference, size error, invalid ID, active padding target, or checksum failure blocks load.
