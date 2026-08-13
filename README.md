# GenPy-200M

GenPy is an educational and research decoder-only language model project intended to be implemented and pretrained from scratch with PyTorch. The architecture and model-level verification are now implemented; training and generation remain later milestones.

## Architecture target

| Property        | Value        |
| --------------- | ------------ |
| Parameters      | 201,560,832 verified |
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

GenPy does not use pretrained model weights. The Transformer implementation uses PyTorch primitives. Hugging Face tooling may later be used for datasets and tokenization, but Hugging Face Transformer model implementations will not be used to build GenPy.

Local CPU execution is intended for development and testing. Kaggle GPU will be used for serious pretraining.

## Roadmap

```text
[x] Step 1 - Project setup
[x] Step 2 - Dataset pipeline
[~] Step 3 - Tokenizer
[x] Step 4 - GenPy architecture
[x] Step 5 - Model verification
[x] Step 6 - Training engine
[x] Step 7 - Small-scale training test
[x] Step 8A - Limited pretraining
[x] Step 8B - Continuation infrastructure prepared
[ ] Step 8B - Continuation training
[ ] Step 9 - Evaluation and inference
[ ] Step 10 - Release
```

Step 1 established the project and environment; later steps add the data, tokenizer, and model components described below.

## Dataset pipeline

Step 2 adds a streaming, resumable, text-only preparation pipeline with conservative cleaning, exact SHA-256 deduplication, deterministic splitting, compressed JSONL shards, manifests, validation, and offline synthetic tests. See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for the workflow and output schema.

## Tokenizer

Step 3 implements and smoke-tests a custom Byte-Level BPE tokenizer with NFC normalization, fixed special-token IDs, explicit BOS/EOS controls, save/load validation, and checksum manifests. Production tokenizer training is pending a real cleaned Step 2 corpus of sufficient size. See [docs/TOKENIZER.md](docs/TOKENIZER.md).

## Architecture

Step 4 implements the locked GenPy-200M decoder-only Transformer: 24
pre-norm blocks, RMSNorm, 12-head causal self-attention with RoPE, SwiGLU
intermediate size 2176, and tied input/output embeddings. The model has a
32,000-token vocabulary and 1,024-token context. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for component details and scope
boundaries.

## Training engine

Step 6 adds streaming uint16 token preparation, memory-mapped packed samples,
deterministic batching, AdamW, warmup/cosine learning rates, precision
management, accumulation, clipping, validation, logging, atomic checkpoints,
and deterministic resume. It builds the engine only; meaningful small-scale
training begins in Step 7. See [docs/TRAINING_ENGINE.md](docs/TRAINING_ENGINE.md).

## Step 7 GPU training

The verified Tesla T4 Step 7 run trained 50 FP16 optimizer steps with sequence
length 256, gradient accumulation 16, and 204,800 tokens. Loss decreased from
10.526969909667969 to 7.654247522354126, with final validation loss
7.657994747161865. Resume from step 45 reproduced steps 46–50 exactly. See
[docs/STEP7_GPU_TRAINING.md](docs/STEP7_GPU_TRAINING.md).

## Step 8A pretraining

Step 8A limited pretraining completed on a Tesla T4 using FP16, sequence length
256, gradient accumulation 16, and 6,297 optimizer steps. The run processed
25,796,608 tokens and reached final loss `4.821587026119232`. See
[docs/STEP8_PRETRAINING.md](docs/STEP8_PRETRAINING.md). Step 8B full pretraining
is not started. The continuation-only initialization design is documented in
[docs/STEP8B_CONTINUATION.md](docs/STEP8B_CONTINUATION.md).

## Verification

Step 5 verifies causal LM loss, finite values and gradients, causal isolation,
RoPE/RMSNorm stability, weight tying, context limits, and tiny-batch learning.
The local CPU verification passes; optional CUDA checks detect unavailable
hardware gracefully. See [docs/VERIFICATION.md](docs/VERIFICATION.md) and the
preserved [GENPY_STEP5_GPU_REPORT.txt](GENPY_STEP5_GPU_REPORT.txt) for recorded
Tesla T4 evidence.
