# GenPy-200M Step 8C Continuation Plan

Step 8C is prepared for substantially larger continuation pretraining, but
training has not started. No 250000-document dataset was downloaded locally
and no GPU run was performed while preparing this plan.

## Checkpoint provenance

Step 8C must initialize from the existing final Step 8B checkpoint:

```text
Source checkpoint: step_00012357.pt
Source completed global step: 12357
Initialization mode: --init-from-checkpoint
```

The initialization is weights-only. It must not restore Step 8B optimizer
moments, scheduler state, FP16 scaler state, sampler position, or RNG state.
The Step 8C run starts fresh phase accounting and a new scheduler. The source
checkpoint SHA-256 must be recorded from the actual artifact before execution;
no new checkpoint artifact is created during preparation.

## Protected and fresh data ranges

The Step 9 unseen FineWeb-Edu evaluation range is permanently protected:

```text
Step 9 evaluation range: source indices [75000, 80000)
Step 8C start index:     80000
Step 8C target documents: 250000
Step 8C intended range:  source indices [80000, 330000)
```

The ranges are disjoint. Step 8C preparation must use
`--skip-documents 80000 --max-documents 250000`. The skip and maximum are
source-row controls; processed-range statistics must describe only the Step
8C selection. The existing manifest/state provenance records the selected
source start and requested range.

The target is expected to provide roughly 250M tokens, but the exact token
count is intentionally not invented here. After tokenization, calculate:

```text
effective tokens per optimizer update = 256 × 1 × 16 = 4096
max_steps = floor(train_token_count / 4096)
```

The final `max_steps` must be supplied only after the exact training token
count is known.

## Step 8C training configuration

Configuration file: `configs/train_step8c_t4.yaml`

```text
seed: 42
sequence_length: 256
micro_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 0.00003
min_learning_rate: 0.000005
weight_decay: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
precision: fp16
beta1: 0.9
beta2: 0.95
adam_eps: 1e-8
log_interval: 20
eval_interval: 500
save_interval: 1000
num_workers: 0
pin_memory: true
eval_batches: 20
keep_last_checkpoints: 3
checkpoint_dir: checkpoints/step8c_t4
log_dir: logs/step8c_t4
```

The target hardware is a Kaggle Tesla T4. The 3e-5 learning rate and 5e-6
minimum are conservative continuation values for learned Step 8B weights.
This file deliberately does not contain a final `max_steps`.

## Bounded Kaggle segments

After tokenization, the exact Step 8C target is expected to be
`max_steps = 60483`. Every invocation must still construct the engine and
scheduler with that full horizon. Long Kaggle runs may be split safely:

```text
Fresh phase:
  --max-steps 60483 --stop-after-steps 5000
  global_step: 0 -> 5000

Resume:
  --max-steps 60483 --stop-after-steps 5000
  global_step: 5000 -> 10000

Final segment:
  --max-steps 60483 --stop-after-steps 5000
  global_step: 60000 -> 60483
```

`stop_after_steps` counts only successful optimizer updates. FP16 overflow
updates remain skipped and do not advance `global_step`, optimizer steps, or
the scheduler. Each bounded invocation saves a complete resumable checkpoint.
Exact `--resume` additionally rejects a checkpoint whose saved `max_steps`
does not equal the requested full horizon, preventing accidental scheduler
schedule changes. `--init-from-checkpoint` remains the separate weights-only
Step 8B-to-Step 8C initialization path.

## Execution gate

When execution is authorized, the data preparation must use the fresh source
range above, validate and tokenize the resulting dataset, record the exact
token count, derive `max_steps`, and initialize with the final Step 8B model
weights using `--init-from-checkpoint`. Exact interrupted-run recovery within
Step 8C may use `--resume` only after Step 8C checkpoints exist.

No Step 8C training, Step 9 benchmark comparison, or artifact publication is
part of this preparation milestone.
