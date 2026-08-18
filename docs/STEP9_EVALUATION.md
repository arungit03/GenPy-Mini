# GenPy-200M Step 9 Evaluation Baseline

This document records the verified Step 9 baseline from an unseen FineWeb-Edu
evaluation slice. It is an evaluation result, not a new training run.

## Verified result

```text
Model: GenPy-200M
Parameters: 201,560,832
Checkpoint step: 12357
Evaluation source: FineWeb-Edu sample-10BT
Evaluation source start: 75000
Source documents examined: 5000
Accepted documents: 4994
Validation documents: 24
Validation tokens: 23299
Sequence length: 256
Complete evaluation windows: 91
Evaluated target tokens: 23296
Validation loss: 4.350139
Perplexity: 77.4892
```

The evaluation slice does not overlap the training data by its configured
source offset and is intended to measure generalization on unseen
FineWeb-Edu documents. The repository evaluator uses the existing
`PackedTokenDataset`, discards only the incomplete tail, computes summed
next-token cross entropy in FP32, and divides by the exact number of target
tokens evaluated.

## Reproducible tooling

Generation is available through `scripts/generate.py` and supports greedy
decoding, temperature sampling, top-k, top-p, repetition penalties, EOS
stopping, seeded sampling, special-token filtering, and model-context
truncation.

Packed validation evaluation is available through `scripts/evaluate.py`:

```text
python scripts/evaluate.py \
  --model-config configs/model_200m.yaml \
  --checkpoint /path/to/step_00012357.pt \
  --validation-data /path/to/validation.bin \
  --validation-metadata /path/to/validation_metadata.json \
  --sequence-length 256 \
  --device cuda \
  --precision fp16 \
  --output evaluation.json
```

Inference loads only the existing `payload["model"]` state dictionary with
strict validation. It does not restore optimizer, scheduler, scaler,
training state, sampler, or training RNG state. Evaluation sets the model to
evaluation mode and runs under `torch.inference_mode()`.

## Comparability requirements

Perplexity comparisons are meaningful only when the tokenizer, token IDs,
sequence length, packed-dataset construction, window policy, and loss
methodology are held constant. Changing any of those can change the reported
value even when the underlying model is unchanged.

The production architecture remains unchanged, including the exact
`201,560,832` parameter count. No tokenizer retraining, model retraining,
checkpoint-format change, or Hugging Face model implementation is part of
Step 9.7.
