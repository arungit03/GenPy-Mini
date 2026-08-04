# Tokenizer Design

GenPy uses a custom byte-level byte-pair encoding (BPE) tokenizer learned only from the
approved GenPy training split. Hugging Face `tokenizers` supplies the implementation, but no
pretrained vocabulary, merges, model weights, or tokenizer files are imported.

## Rationale

Byte-level BPE gives complete UTF-8 byte coverage without an unknown-token fallback. It can
represent Python, prose, Tamil, emoji, paths, and unusual valid Unicode while still learning
multi-byte fragments that compress common text. Byte coverage guarantees representability;
it does not guarantee good compression, identifier boundaries, or model quality.

The production vocabulary is fixed at 16,384 entries, including seven control tokens. This
keeps embedding and output layers practical for the approximately 5M, 25M, and 100M model
scales while leaving room for useful Python and natural-language fragments. All model scales
must use one frozen tokenizer so IDs and checkpoints remain compatible.

Python indentation, tabs, blank lines, and line endings are semantic or stylistically
important, so training and encoding preserve them exactly. Unicode normalization,
lowercasing, formatting, and prefix-space insertion are disabled. The byte-level decoder must
reproduce the input exactly.

## Locked Contract

| ID | Token |
| ---: | --- |
| 0 | `<|pad|>` |
| 1 | `<|bos|>` |
| 2 | `<|eos|>` |
| 3 | `<|user|>` |
| 4 | `<|assistant|>` |
| 5 | `<|code|>` |
| 6 | `<|end|>` |

Pretraining records use:

```text
<|bos|><|code|>
{python_source}<|end|><|eos|>
```

Instruction records use:

```text
<|bos|><|user|>
{user_prompt}<|assistant|><|code|>
{assistant_python_code}<|end|><|eos|>
```

No padding is added to individual records. Structural insertion is separate from ordinary
text encoding, and no truncation occurs in the tokenizer wrapper.

## Collision Policy

Any exact reserved-token string in source code, prompts, or answers is rejected. It is never
silently interpreted as a control boundary. A future reversible escaping policy would require
a new documented format and tests before use.

## Compatibility

The tokenizer fingerprint covers the vocabulary, merges, tokenizer settings, control-token
map, corpus fingerprint, and configuration hash. Changing any of these creates a new tokenizer
version. Once model training begins, the referenced tokenizer is immutable; changing it
invalidates checkpoint embedding and output IDs.

## Limitations

The current repository contains only a one-repository smoke corpus. Its 1,024-entry smoke
tokenizer verifies infrastructure but is not suitable for model training. Byte-identical
reproducibility has been verified twice on the current Windows environment only, not across
Windows, Linux, and Kaggle. Static vocabulary scanning cannot guarantee that all sensitive
fragments are absent.
