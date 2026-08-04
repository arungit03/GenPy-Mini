# ADR-004: Tokenizer And Serialization

Status: accepted for Phase 3; production artifact not yet frozen.

## Context

GenPy needs stable IDs for Python source and natural-language instruction data without relying
on any existing language-model tokenizer. Whitespace and arbitrary valid UTF-8 must survive a
round trip, and all three model scales must share one vocabulary.

## Decision

GenPy will train a custom byte-level BPE from an empty model using only approved Phase 2 train
records. The production vocabulary is exactly 16,384 entries. IDs 0 through 6 are permanently
assigned to `<|pad|>`, `<|bos|>`, `<|eos|>`, `<|user|>`, `<|assistant|>`, `<|code|>`, and
`<|end|>` in that order.

Pretraining serialization is `<|bos|><|code|>\n{python_source}<|end|><|eos|>`. Instruction
serialization is `<|bos|><|user|>\n{prompt}<|assistant|><|code|>\n{code}<|end|><|eos|>`.
The fixed GenPy system message is validated and omitted. No Markdown fences or padding are
inserted.

Literal reserved-token strings in content are rejected. No silent control-token
interpretation or undocumented escaping is allowed.

Vocabulary and merges may learn only from training splits. Validation, test, private
benchmarks, quarantined/rejected data, and test fixtures are excluded. Artifacts are versioned
and fingerprinted. Production files cannot be overwritten under the same version.

Once any model training begins, the tokenizer becomes immutable. Checkpoints must store the
fingerprint and future loaders must reject a mismatch.

## Consequences

All valid UTF-8 remains representable and Python whitespace is preserved. Tokenizer quality
still depends on corpus scale and balance. Any corpus, serialization, vocabulary, merge, or
control-token change requires a new tokenizer version and invalidates existing checkpoints.
