# GenPy-200M Step 7 GPU Training

Step 7 performed a controlled small-scale GPU training run using the completed
Step 6 engine. Step 8 was not started.

## Verified run

```text
Model: GenPy-200M
Parameters: 201,560,832
GPU: Tesla T4
Precision: FP16
Sequence length: 256
Micro batch size: 1
Gradient accumulation: 16
Effective tokens/update: 4096
Optimizer steps: 50
Tokens trained: 204,800
```

## Metrics

```text
Initial loss: 10.526969909667969
Final loss: 7.654247522354126
Final validation loss: 7.657994747161865
```

The run used the real FineWeb-Edu subset, the production 32K tokenizer, and
the packed uint16 dataset. FP16 autocasting, gradient accumulation, gradient
clipping, the learning-rate scheduler, validation, checkpoint saving,
retention, and checkpoint integrity were verified.

## Resume determinism

Resume from checkpoint step 45 reproduced the exact training trajectory for
steps 46–50, including:

- training losses;
- learning-rate values; and
- gradient norms.

## Artifact hashes

The following SHA-256 values are the verified hashes supplied for the Step 7
artifacts:

```text
Final checkpoint:
0670ae42d6b1f7628e2aa536cd80279f628c47637d623cd2d4197756d70493ea

Production tokenizer:
4f9eb931708775d5ac1e1a3fbf0105e3b9bfaf2477ebe0a1463e9a7fc3272fae
```

These values are recorded evidence from the verified GPU run. No replacement
or fabricated local artifact is claimed here.

## Cleanup verification

The training engine now:

1. logs cumulative `TrainingState.tokens_seen` after checkpoint resume;
2. restores validation sampler state after each validation call; and
3. preserves exact deterministic resume behavior.

The production architecture remains unchanged at exactly 201,560,832
parameters.

## Boundary

Step 8 was not implemented or started. No architecture changes, tokenizer
retraining, or full pretraining were performed as part of this cleanup.
