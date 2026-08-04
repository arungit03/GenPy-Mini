# Token Packing

Production samples store 1,025 token IDs. Inputs are positions `0:1024` and labels are positions
`1:1025`, giving exactly 1,024 next-token targets without a second model-side shift. The smoke
pipeline applies the same rule at width 65 for its 64-token context.

Packing reuses the Phase 3 tokenizer wrapper and canonical serialization. It never formats code,
adds fences, changes indentation, truncates records, or introduces special tokens. Long records
continue across blocks. A stride equal to context length leaves the final token of one stored
sample as the first input of the next, so targets are not skipped or duplicated.

Families and splits are packed independently. Records may share a block, and ordinary causal
attention may see prior records. EOS and BOS identify boundaries. The EOS-to-next-BOS target is
masked by default; block-diagonal attention is not claimed.

Pretraining uses `full_lm`: all valid next-token targets except padding and cross-record BOS
transitions are active. Instruction data uses `assistant_only`: prompt tokens remain visible,
while loss starts on `<|assistant|>` and includes `<|code|>`, assistant code, `<|end|>`, and
`<|eos|>`. Padding targets are always inactive.

The rolling buffer is bounded by one output shard. Source shards and checksums are processed in
stable order. Existing matching output is resumable as a completed unit; incomplete replacement
requires explicit smoke `--force`. Production output cannot be replaced in place. Repeated smoke
packing produced byte-identical token and mask binaries.
