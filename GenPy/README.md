# GenPy-200M

GenPy is an educational and research decoder-only language model project intended to be implemented and pretrained from scratch with PyTorch. This repository is currently at Step 1: project setup and environment preparation.

## Architecture target

| Property        | Value        |
| --------------- | ------------ |
| Parameters      | ~200M target |
| Layers          | 24           |
| Hidden Size     | 768          |
| Attention Heads | 12           |
| Head Dimension  | 64           |
| FFN             | SwiGLU 2176  |
| Vocabulary      | 32K          |
| Context         | 1024         |
| Position        | RoPE         |
| Normalization   | RMSNorm      |
| Framework       | PyTorch      |

GenPy will not use pretrained model weights. The future Transformer implementation will use PyTorch primitives. Hugging Face tooling may later be used for datasets and tokenization, but Hugging Face Transformer model implementations will not be used to build GenPy.

Local CPU execution is intended for development and testing. Kaggle GPU will be used for serious pretraining.

## Roadmap

```text
[x] Step 1 - Project setup
[x] Step 2 - Dataset pipeline
[~] Step 3 - Tokenizer
[ ] Step 4 - GenPy architecture
[ ] Step 5 - Model verification
[ ] Step 6 - Training engine
[ ] Step 7 - Small-scale training test
[ ] Step 8 - Full pretraining
[ ] Step 9 - Evaluation and inference
[ ] Step 10 - Release
```

Step 1 intentionally contains no model, tokenizer, dataset, or training implementation.

## Dataset pipeline

Step 2 adds a streaming, resumable, text-only preparation pipeline with conservative cleaning, exact SHA-256 deduplication, deterministic splitting, compressed JSONL shards, manifests, validation, and offline synthetic tests. See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for the workflow and output schema.

## Tokenizer

Step 3 implements and smoke-tests a custom Byte-Level BPE tokenizer with NFC normalization, fixed special-token IDs, explicit BOS/EOS controls, save/load validation, and checksum manifests. Production tokenizer training is pending a real cleaned Step 2 corpus of sufficient size. See [docs/TOKENIZER.md](docs/TOKENIZER.md).
