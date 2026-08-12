# GenPy tokenizer

Step 3 implements the GenPy tokenizer as a standalone Hugging Face `tokenizers` Byte-Level BPE. It does not use a pretrained tokenizer or `transformers`.

## Configuration

The production configuration is in `configs/tokenizer.yaml`:

- algorithm: Byte-Level BPE
- vocabulary: 32,000 including special tokens
- normalizer: NFC
- ByteLevel `add_prefix_space: false`, regex enabled
- minimum pair frequency: 2
- maximum token length: 64
- initial alphabet: all ByteLevel byte representations

The model/tokenizer contract requires both vocabulary values to be 32,000.

## Special tokens

The first four IDs are fixed:

| ID | Token |
|---:|---|
| 0 | `<|pad|>` |
| 1 | `<|bos|>` |
| 2 | `<|eos|>` |
| 3 | `<|unk|>` |

Raw encoding does not add BOS or EOS. `GenPyTokenizer.encode` exposes explicit `add_bos` and `add_eos` flags. `encode_document` appends EOS only. Padding, truncation, and sequence packing are not implemented in Step 3.

## Training corpus and modes

Training reads Step 2 `train-*.jsonl.gz` shards incrementally and never concatenates the corpus into a Python list. Validation shards are reserved for evaluation. Corpus statistics include documents, characters, and UTF-8 bytes.

The smoke command uses the local synthetic fixture and a clearly non-production 512-token vocabulary:

```bash
python scripts/train_tokenizer.py --smoke
```

Production training requires a completed Step 2 manifest and real cleaned train data. The project currently has no real Step 2 shards because the earlier FineWeb-Edu access was blocked by the environment's Hugging Face SSL certificate failure. Therefore no production tokenizer is claimed.

The production gate is 256 MiB minimum cleaned real text, with an approximately 1 GiB target. Synthetic data must never be used to label the final tokenizer production-ready.

## Artifacts and validation

The canonical artifact is `data/tokenizer/genpy-tokenizer.json`. Training also records configuration and a manifest containing corpus provenance, settings, versions, counts, mode, and a SHA-256 checksum. The validator checks JSON validity, vocabulary, special IDs, model vocabulary compatibility, NFC/ByteLevel settings, round-trip behavior, and checksum.

Useful commands:

```bash
python scripts/inspect_tokenizer.py --tokenizer path/to/genpy-tokenizer.json --text "GenPy café தமிழ்"
python scripts/evaluate_tokenizer.py --tokenizer path/to/genpy-tokenizer.json --input-dir data/processed
python scripts/verify_tokenizer.py --tokenizer path/to/genpy-tokenizer.json --manifest path/to/tokenizer_manifest.json
```

## Evaluation and limitations

The diagnostic suite covers English prose, Python/C-like code, numbers, mathematics, URLs, Tamil, Hindi, emoji, and mixed-language text. Byte-level coverage is representation capability, not evidence of multilingual language quality. Tokenizer quality depends on the real cleaned training corpus; the current implementation is verified offline with smoke data, while production verification remains pending.
