# GenPy Tokenizer

GenPy-Tokenizer-32K is a custom Byte-Level BPE tokenizer trained from scratch for Python instructions and Python source. It does not reuse a pretrained model vocabulary, tokenizer JSON, or merge table.

## Training contract

Vocabulary learning uses only `data/instruction/python/train.jsonl` (90,000 examples). Each document is formatted with registered `<BOS>`/`<EOS>` boundaries, `### User`, the instruction, optional input, `### Assistant`, and the response. IDs, hashes, family labels, quality fields, and source metadata are not corpus text. The production train file is verified against the Checkpoint 2.5 SHA-256 before training.

The tokenizer uses byte-level pre-tokenization with `add_prefix_space: false`, no lowercasing, no Unicode normalization, no indentation stripping, and no whitespace collapsing. UTF-8 bytes are preserved through encode/decode, including spaces, tabs, newlines, punctuation, operators, underscores, and Unicode.

The production corpus contains fewer than 32,000 distinct eligible BPE merge entries. The artifact therefore records its composition explicitly: 13,271 entries come from the learned BPE model and 18,729 additional vocabulary entries are deterministic, observed substrings from the same train-only corpus. No external vocabulary or merge table is used.

## Contract

| Item | Value |
|---|---|
| Type | Byte-Level BPE |
| Vocabulary | 32,000 |
| `<PAD>` | 0 |
| `<BOS>` | 1 |
| `<EOS>` | 2 |
| `<UNK>` | 3 |

`GenPyTokenizer.load(path)` loads a local artifact. `encode(text, add_bos=False, add_eos=False)` returns IDs, and `decode(ids, skip_special_tokens=False)` returns text. With `skip_special_tokens=False`, registered specials are emitted literally; with it enabled, they are omitted. The wrapper also provides `encode_batch`.

## Artifacts and reproducibility

The production artifact is under `artifacts/tokenizer/genpy-32k/` and includes `tokenizer.json`, `vocab.json`, `merges.txt`, configuration, special-token metadata, and a manifest. Reports record corpus statistics, file hashes, split metrics, and the tokenizer library/Python versions.

Rebuild with:

```powershell
python scripts/train_tokenizer.py --config configs/tokenizer.yaml --input data/instruction/python/train.jsonl --output artifacts/tokenizer/genpy-32k
```

Run the validation, inspection, and benchmark scripts after training. The 0% UNK objective follows from the complete byte alphabet; it is still measured explicitly on train, validation, test, and synthetic Unicode examples.

## Limitations

This checkpoint does not implement model packing, padding collation, embeddings, a Transformer, or training. Token count is an efficiency diagnostic, not a standalone measure of tokenizer quality. A future tokenizer version must be versioned rather than silently replacing this artifact.
