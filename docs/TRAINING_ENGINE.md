# GenPy-200M Step 6 Training Engine

Step 6 builds a modular, single-process training engine. It does not start
meaningful GenPy-200M pretraining; that belongs to Step 7.

## Tokenized data

Cleaned Step 2 JSONL.GZ shards are streamed through the verified Step 3
tokenizer. Each document is encoded and terminated with EOS. The resulting
`uint16` token stream is written atomically beside JSON metadata. Vocabulary
32000 fits safely in `uint16`, and metadata records the tokenizer checksum,
special IDs, source shards, counts, and dtype.

## Packed samples

`PackedTokenDataset` memory-maps the binary file. A sample reads
`sequence_length + 1` contiguous IDs and returns:

```text
input_ids = tokens[:-1]
targets   = tokens[1:]
```

Samples may cross document boundaries because EOS is an explicit token. No PAD
tokens are inserted and only an unusable final tail is discarded.

## DataLoader and determinism

`StatefulBatchSampler` uses a seed and stores epoch plus batch position. It is
single-process and works with `num_workers=0`, which is the deterministic
resume configuration for Step 6.

## Optimizer and schedule

The engine uses PyTorch `AdamW`. Parameters with dimension at least two receive
weight decay; vectors such as RMSNorm scales do not. Shared tied parameters are
deduplicated before grouping.

Learning rate is updated once per optimizer step using linear warmup followed
by cosine decay toward `min_learning_rate`. Gradient-accumulation microsteps do
not advance the scheduler.

## Gradient handling

Each microbatch loss is divided by `gradient_accumulation_steps`. At the
boundary, FP16 gradients are unscaled, finite gradients are required, optional
`clip_grad_norm_` is applied, and only then does the optimizer and scheduler
advance. A `grad_clip` value of zero disables clipping.

## Precision

`PrecisionManager` supports `fp32`, CUDA `fp16`, CUDA `bf16`, and `auto`.
CPU automatically uses FP32; explicit unsupported CUDA precision requests fail
clearly. FP16 uses GradScaler, while BF16 does not require one.

## Validation and metrics

Validation uses `model.eval()` and `torch.no_grad()` for a configured number of
batches, averages batch losses, and restores the prior training mode. Metrics
include loss, learning rate, gradient norm, steps, tokens, throughput, elapsed
time, and guarded perplexity. Console and JSONL logging are provided without
cloud dependencies.

## Checkpoints and resume

Checkpoints contain model, optimizer, scheduler, precision/scaler, training
state, sampler state, model/training configuration, data metadata, and Python,
NumPy, Torch, and CUDA RNG states. Files are written to a temporary path and
atomically renamed. A small `latest.json` pointer identifies the latest valid
checkpoint, and retention preserves the configured number of recent files.

Only trusted local checkpoint paths should be loaded. Step 6 does not download
or load pretrained weights.

## CLI and smoke test

`scripts/train.py` requires explicit `--max-steps`, validates data/config/model
before updates, reports effective sequences/tokens per optimizer update, and
supports `--dry-run`. `scripts/training_smoke_test.py` exercises data, forward,
loss, accumulation, clipping, optimizer, scheduler, logging, validation-ready
engine construction, checkpointing, and deterministic resume on CPU.

## Step 7 boundary

Step 6 does not implement distributed training, DDP, FSDP, DeepSpeed, full
FineWeb-Edu preparation, production pretraining, generation, sampling, or
instruction tuning. Step 7 will use this engine for a controlled small-scale
GPU training run.
