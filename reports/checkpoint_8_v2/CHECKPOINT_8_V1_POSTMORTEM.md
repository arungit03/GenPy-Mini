# Checkpoint 8 v1 postmortem

The original Checkpoint 8 SFT plan was not promoted to production. Its 90,000/5,000/5,000 dataset remains unchanged, but the legacy test split is classified as `LEGACY_IN_DISTRIBUTION_CONTAMINATION_RISK`: train/test prompt overlap was 0, while normalized solution overlap was 699 and template overlap was 51. Checkpoint 7 pretraining also used the production training split, so the legacy test is not an unseen-family generalization result.

The v1 budget also initially described static token capacity as effective tokens/update. That was insufficient for response-only loss because prompt and padding positions do not contribute supervised loss. The v2 review reports formatted tokens, assistant-supervised tokens, ignored prompt tokens, padding, length percentiles, and actual per-update estimates separately.

The v2 pilot is independently generated and uses separate paths. Its frozen challenge set is excluded from SFT and hyperparameter selection, and its frozen sanity set is evaluation-only. The v2 schedule is fixed at 2,500 global optimizer steps before any health gate; session limits cannot change scheduler `total_steps`.

The local workspace does not contain the immutable Checkpoint 7 `model.pt` with trusted SHA256 `a963a91d8f6bee350e15ff88d3375c039887cb0b09c787fecf0f2de02d5be942`, and CUDA is unavailable. Therefore no production optimizer step was run and the final readiness status remains not ready locally.
