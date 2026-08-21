# GenPy Checkpoint 7 Pretraining

The canonical production configuration is
`configs/pretrain_200m_kaggle.yaml`. It fixes the global scheduler budget at
1,980 optimizer updates and must remain unchanged across Kaggle sessions.

## Preflight and first session

Run these commands on the Kaggle GPU runtime:

```text
python scripts/pretrain_checkpoint7_preflight.py
python scripts/train_genpy.py --model-config configs/model_200m.yaml --train-config configs/pretrain_200m_kaggle.yaml --data data/tokenized/genpy-32k/TOKEN_CACHE_MANIFEST.json --run-dir runs/genpy200m_pretrain_v1 --session-steps 100
```

The first invocation is the 100-step health gate. `--session-steps` limits
additional optimizer updates in that process only; it never changes
`max_steps`, scheduler `total_steps`, run ID, or token accounting. A bounded
session ends with `SESSION_COMPLETE` and saves a checkpoint. Reaching step
1,980 prints `TRAINING_COMPLETE`.

Resume the same run with:

```text
python scripts/train_genpy.py --model-config configs/model_200m.yaml --train-config configs/pretrain_200m_kaggle.yaml --data data/tokenized/genpy-32k/TOKEN_CACHE_MANIFEST.json --run-dir runs/genpy200m_pretrain_v1 --resume auto --session-steps 400
```

## Monitoring and Kaggle transport

Metrics append to `runs/genpy200m_pretrain_v1/logs/training_metrics.jsonl`.
Inspect them with:

```text
python scripts/inspect_pretraining_progress.py
```

Because `/kaggle/working` is temporary, package the latest complete checkpoint
before leaving a session:

```text
python scripts/package_latest_checkpoint.py --run-dir runs/genpy200m_pretrain_v1 --output /kaggle/working/genpy200m_latest_checkpoint.tar
```

Restore it in a later session with `restore_checkpoint_archive.py` and verify
the accompanying `.sha256` file. The archive contains model/optimizer state,
RNG and scheduler state, config snapshots, manifests, and metrics; it never
contains the token cache or old checkpoints.
