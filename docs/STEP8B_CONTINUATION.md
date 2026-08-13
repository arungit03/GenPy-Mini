# GenPy-200M Step 8B Continuation Infrastructure

Step 8B continuation infrastructure is prepared, but training has not
started. No training command was run while preparing this support.

## Initialization modes

```text
--resume PATH
    Exact interrupted-run recovery. Restores model, optimizer, scheduler,
    FP16 scaler, TrainingState, sampler, RNG state, and compatible metadata.

--init-from-checkpoint PATH
    New continuation phase. Loads model weights only and starts fresh phase
    accounting, optimizer, scaler, scheduler, sampler, and RNG state.
```

The modes are mutually exclusive. A fresh Step 8B dataset is never paired with
the Step 8A sampler position by the continuation path.

## Reset policy

The default continuation policy resets optimizer moments and the FP16 scaler.
This is the cleaner default because Step 8B uses fresh data and a new
warmup/cosine schedule. Carrying moments or loss-scale history from the prior
phase would silently mix optimizer state across a new data/schedule boundary.

The Step 8A model weights are retained exactly. Exact crash recovery remains
available through `--resume`, which restores all state components.

## Provenance

Continuation checkpoints record the source checkpoint path, source SHA-256,
source completed optimizer step, source tokens seen, and
`initialization_mode: weights_only`. They also record that optimizer,
scheduler, scaler, and sampler state were not restored from the source.

## Configuration skeleton

`configs/train_step8b_t4.yaml` uses conservative continuation values:

```text
sequence_length: 256
micro_batch_size: 1
gradient_accumulation_steps: 16
precision: fp16
learning_rate: 5e-5
min_learning_rate: 1e-5
weight_decay: 0.1
grad_clip: 1.0
save/eval interval: 250 steps
logging interval: 10 steps
```

The `5e-5` LR is deliberately below the Step 8A peak `3e-4`; the reduced
`1e-5` floor and 2% warmup provide a conservative new schedule for learned
weights. These values are preparation only, not training validation.

## Example, not executed

```text
python scripts/train.py \
  --model-config configs/model_200m.yaml \
  --train-config configs/train_step8b_t4.yaml \
  --train-data PATH_TO_FRESH_TRAIN_BIN \
  --validation-data PATH_TO_FRESH_VALIDATION_BIN \
  --checkpoint-dir checkpoints/step8b_t4 \
  --log-dir logs/step8b_t4 \
  --device cuda \
  --max-steps EXPLICIT_STEP8B_LIMIT \
  --init-from-checkpoint checkpoints/step8a_full/step_00006297.pt
```

This command is documentation only and was not run. Step 8B training remains
not started.

## Safety guarantees

- Model configuration compatibility is checked before weight initialization.
- Exact resume still rejects incompatible sampler/data state.
- The new scheduler begins at Step 8B's configured LR and step zero.
- Step 8B global step, token, microstep, and optimizer-step counters begin at zero.
- The production model remains exactly 201,560,832 parameters.
- The tokenizer and data pipeline are unchanged.
