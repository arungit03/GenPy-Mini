# GenPy-200M Step 8A Pretraining

Step 8A is the verified limited-pretraining milestone for GenPy-200M. It is
not final 200M-model pretraining completion, and Step 8B has not started.

## Production model

```text
Model: GenPy-200M
Parameters: 201,560,832
Vocabulary: 32,000
Architecture: unchanged decoder-only Transformer
```

The model architecture and parameter count were preserved throughout Step 8A.

## Hardware and training configuration

```text
Hardware: Kaggle Tesla T4
Precision: FP16 with torch GradScaler
Sequence length: 256
Micro batch size: 1
Gradient accumulation: 16
Effective tokens/update: 4096
Learning rate: 3e-4
Minimum learning rate: 3e-5
Weight decay: 0.1
Gradient clipping: 1.0
Target optimizer steps: 6,297
```

## Dataset and tokenizer

The run used the cleaned FineWeb-Edu subset and the existing production
tokenizer. No tokenizer retraining or data-pipeline change was performed.

```text
Train documents: 24,865
Validation documents: 105
Train tokens: 25,796,499
Validation tokens: 111,027
Vocabulary: 32,000
Packed format: uint16
```

## 100-step sanity run

The sanity run completed 100 optimizer steps and processed 409,600 tokens.

```text
Step 25:  7.654693365097046
Step 50:  7.348004102706909
Step 75:  7.196495771408081
Step 100: 7.1258544921875
Result: PASS
```

## Full Step 8A run

The limited run completed all 6,297 target optimizer steps successfully after
resuming from the safe step-6000 checkpoint.

```text
Completed optimizer steps: 6,297
Final tokens_seen: 25,796,608
Final learning rate: 3e-5
Late-run loss: approximately 4.7–4.9
Final loss: 4.821587026119232
Result: PASS
```

The final token count is consistent with the configured 4,096 effective tokens
per optimizer update over 6,297 updates.

## FP16 overflow incident and recovery

The original run encountered non-finite gradients near `global_step=6242`.
The training engine was updated to handle FP16 GradScaler overflow correctly:

- detect non-finite gradients after unscale;
- skip the affected optimizer update;
- reduce/update the GradScaler;
- clear gradients;
- do not advance the learning-rate scheduler;
- do not increment `global_step` or `optimizer_steps`;
- continue counting consumed microsteps and tokens; and
- emit an explicit overflow/skipped-update log record.

FP32 and BF16 retain strict failure behavior for non-finite gradients.

Training resumed from the safe step-6000 checkpoint and completed through
step 6297. The full post-fix test suite passed with 102 tests passed, 1 skipped,
and 0 failures.

## Checkpoint and resume

```text
Final checkpoint:
checkpoints/step8a_full/step_00006297.pt

Checkpoint SHA-256:
c85aa1aa246faac5fd309b48b2ec87c95457588c0c7aec59e0f56aa2b283dfca
```

The final 2.3 GB checkpoint was reported as downloaded successfully. The hash
above is the verified artifact hash supplied for the completed Step 8A run.
Resume from step 6000 completed successfully after the FP16 overflow handling
fix.

## Tokenizer artifact

```text
Tokenizer SHA-256:
4f9eb931708775d5ac1e1a3fbf0105e3b9bfaf2477ebe0a1463e9a7fc3272fae
```

The production 32K tokenizer and its contract were preserved unchanged.

## Limitations

Step 8A is intentionally limited:

- approximately 25.8M training tokens were processed;
- sequence length was 256 rather than the 1,024 production context;
- this is not full-scale pretraining for a 200M-parameter model;
- no claim of final language-model quality or release readiness is made; and
- the run is a controlled pretraining milestone before later work.

## Step boundary

Step 8A documentation and verification are complete. Step 8B is **not
started**. This cleanup did not retrain anything, modify the architecture,
change the tokenizer, change the data pipeline, or implement unrelated Step 8
features.
