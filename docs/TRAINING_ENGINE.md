# GenPy Training Engine

Checkpoint 6 provides a resumable, single-process PyTorch training engine for
the native GenPy-200M model. It is designed for a single Kaggle GPU, while
remaining fully testable on CPU with a tiny verification model.

## Data path

`scripts/build_token_cache.py` reads only the approved train and validation
JSONL files. Each document is formatted as a user/assistant conversation,
encoded with the custom 32K tokenizer, and receives BOS and EOS IDs
programmatically. The resulting `uint16` streams are stored as memory-mapped
`train.bin` and `validation.bin` files. A dataset window of `sequence_length +
1` tokens produces causal inputs `x=tokens[:-1]` and targets `y=tokens[1:]`.

Build and inspect the production cache with:

```text
python scripts/build_token_cache.py
python scripts/inspect_token_cache.py
```

The cache manifest records source hashes, tokenizer identity, token counts,
special-token counts, and binary hashes. Test data is never included.

## Configuration and safety

`configs/training_engine.yaml` is the production default. A run must provide
exactly one of `training.max_steps` or `training.max_tokens`; a missing budget
is accepted for inspection/dry-run but refuses to start training. The engine
supports `auto`, CPU FP32, CUDA BF16, and CUDA FP16 with GradScaler.

The optimizer is AdamW with betas `(0.9, 0.95)`, epsilon `1e-8`, and weight
decay `0.1`. Embeddings, normalization parameters, biases, and the tied output
weight are kept in the no-decay group. The scheduler applies linear warmup and
cosine decay per optimizer update.

## Resume and checkpoints

Checkpoints are written to a temporary directory and atomically renamed only
after required files and a `COMPLETE` marker are present. They contain model,
optimizer, scheduler, scaler, trainer state, metadata, RNG state, and batcher
state. Resume compatibility checks cover model/tokenizer/cache identity and
the optimizer state is moved to the active device before continuing.

Useful commands:

```text
python scripts/train_genpy.py --config configs/training_engine.yaml --dry-run
python scripts/train_genpy.py --config configs/training_smoke.yaml
python scripts/run_training_smoke.py
python scripts/validate_training_engine.py
```

The local smoke validates loss/backpropagation, clipping, checkpointing,
validation, RNG restoration, and exact interrupted/resumed equivalence. The
production CUDA smoke remains pending until a GPU/Kaggle runtime is available.
