# Model Configuration

`src/genpy/model/config.py` accepts only the documented model, tokenizer, and training keys.
Unknown or missing keys fail before tensor allocation. It validates positive dimensions,
head divisibility, even RoPE dimensions, standard MHA, dropout ranges, supported components,
bias-free layers, tied embeddings, fixed production context/vocabulary, and locked special IDs.

The production configs retain their Phase 1 architecture dimensions and reference the same
future `genpy-byte-bpe-16k` artifact. Their fingerprint remains
`populated_after_training` because no production tokenizer exists. The separate smoke config
references the validated 1,024-token smoke artifact and must never be used for model training.

Useful commands:

```powershell
python scripts/model/check_readiness.py --config configs/model/genpy_5m.yaml
python scripts/model/count_parameters.py --config configs/model/genpy_5m.yaml
python scripts/model/estimate_memory.py --config configs/model/genpy_100m.yaml --sequence-length 1024 --micro-batch-size 1
```

Parameter counting is allocation-free and does not double-count tied weights. Configuration
changes alter the hash and make existing checkpoints incompatible unless explicitly migrated as
a new architecture version.
