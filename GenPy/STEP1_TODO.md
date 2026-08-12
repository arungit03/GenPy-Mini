# GenPy-200M — STEP 1 TODO

Use these states: `[ ]` Not started, `[~]` In progress, `[x]` Completed and verified, `[!]` Blocked/problem found.

## TODO 1 — Inspect Current Repository
- [x] Inspect the current working directory, existing files, GenPy presence, useful files, and project root.
- Verification: repository state understood before project creation.

## TODO 2 — Create Project Structure
- [x] Create the required GenPy directories/files and package `__init__.py` files.
- Verification: every required directory/file exists and package directories have `__init__.py`.

## TODO 3 — Create requirements.txt
- [x] Add exactly `torch`, `numpy`, `pyyaml`, `tqdm`, `tokenizers`, `datasets`, `safetensors`, `tensorboard`, and `pytest`; do not add `transformers` or unnecessary dependencies.
- Verification: inspect requirements and confirm all nine dependencies and no forbidden package.

## TODO 4 — Create pyproject.toml
- [x] Create minimal professional TOML with project name `genpy`, Python requirement, metadata, pytest configuration, and clean local package import support.
- Verification: TOML syntax and package naming are valid.

## TODO 5 — Create Model Configuration
- [x] Create `configs/model_200m.yaml` with the exact GenPy-200M model configuration.
- Verification: verify `768 % 12 == 0`, `768 / 12 == 64`, and all values match.

## TODO 6 — Create Training Configuration
- [x] Create `configs/train.yaml` with the exact specified training values; do not implement a trainer.
- Verification: YAML loads, `training` exists, fields exist, and numeric values are valid.

## TODO 7 — Implement Configuration System
- [x] Implement simple `ModelConfig` and `TrainingConfig` dataclasses in `genpy/config.py` with YAML loading, required-field checks, useful errors, numeric validation, and dimension relationships. Do not implement architecture classes.
- Verification: load both configuration files successfully.

## TODO 8 — Implement Device Helper
- [x] Create `genpy/utils/device.py` returning CUDA when available and CPU otherwise, without hardcoding CUDA.
- Verification: run it and confirm a valid PyTorch device.

## TODO 9 — Implement Random Seed Utility
- [x] Create `genpy/utils/seed.py` seeding Python random, NumPy, PyTorch CPU, and available CUDA generators without forcing expensive deterministic algorithms.
- Verification: execute successfully with and without CUDA.

## TODO 10 — Create Environment Checker
- [x] Create `scripts/check_environment.py` reporting Python, PyTorch, CUDA, cuDNN, device count, GPU name/VRAM, BF16, FP16, and selected GenPy device. Handle unsupported APIs safely and allow CPU-only mode.
- Verification: run the script successfully.

## TODO 11 — Create .gitignore
- [x] Ignore Python caches, environments, secrets, generated data, checkpoints, logs, and notebook checkpoints while preserving required `.gitkeep` placeholders.
- Verification: generated artifacts are ignored and placeholders are trackable.

## TODO 12 — Create README.md
- [x] Create the professional README with title, architecture table, project constraints, roadmap, and `[~] Step 1 - Project setup` until final verification.
- Verification: update Step 1 to `[x]` only after every preceding TODO passes.

## TODO 13 — Create Kaggle Preparation Documentation
- [x] Create `notebooks/README.md` documenting Kaggle import/clone, GPU enablement, requirements installation, environment checker, CUDA verification, and later notebooks. Do not create a training notebook or download data.

## TODO 14 — Create Configuration Tests
- [x] Create `tests/test_config.py` covering valid loads, dimensions, expected values, invalid dimensions/values, and missing sections.

## TODO 15 — Create Utility Tests
- [x] Create `tests/test_utils.py` covering device helper, CPU fallback where practical, seed execution, and reproducibility. Do not test model functionality.

## TODO 16 — Create Parameter Estimation Script
- [x] Create `scripts/count_parameters.py` with a clearly labeled theoretical estimate derived from configuration and documented assumptions. State that exact counting is unavailable until the model exists; do not instantiate another model.

## TODO 17 — Import Verification
- [x] Verify `genpy`, configuration classes, and Step 1 utilities import cleanly without runtime `sys.path` hacks.

## TODO 18 — Run Environment Verification
- [x] Run `python scripts/check_environment.py`, check exit status, and record actual results. CPU-only operation is acceptable.

## TODO 19 — Run Complete Test Suite
- [x] Run `pytest`; investigate and fix failures before marking complete.

## TODO 20 — Inspect Project for Scope Violations
- [x] Search for forbidden Step 2+ implementations: model, attention, RoPE, RMSNorm, SwiGLU, tokenizer training, datasets/downloads, training loop, pretrained loading, GPT2/Llama model classes. Confirm none were created.

## TODO 21 — Verify Git Safety
- [x] Confirm raw/processed/tokenizer data, checkpoints, logs, caches, and environment files are ignored; `.gitkeep` placeholders remain trackable; no user files were deleted. Git status also showed pre-existing deletions outside `GenPy/`; those files were not modified by Step 1.

## TODO 22 — Final File Tree Audit
- [x] Inspect the final tree and compare it with the required structure; fix missing files.

## TODO 23 — Final Configuration Audit
- [x] Re-open both YAML files and confirm all GenPy-200M and training values remain exact, including `768 / 12 = 64`.

## TODO 24 — Final Test
- [x] Run `pytest` again and `python scripts/check_environment.py` again; fix any failure and repeat.

## TODO 25 — Complete README Roadmap
- [x] After all preceding TODOs pass, change `[~] Step 1 - Project setup` to `[x] Step 1 - Project setup` and do not start Step 2.

## TODO 26 — Final TODO Audit
- [x] Read this file from beginning to end and calculate Total, Completed, Remaining, and Blocked. Required final state: Remaining `0`, Blocked `0`.

## Strict Scope Boundary
Do not implement multi-head/causal attention, RoPE, RMSNorm, SwiGLU, Transformer blocks, the GenPy model, tokenizer training, dataset pipeline/download/tokenization, optimizer, scheduler, training loop, gradient accumulation, mixed precision training, checkpoint manager, text generation, or a full parameter counter. Do not use GPT2/Llama/AutoModel model shortcuts or pretrained weights.

## Failure Handling and Final Questions
Every failed command must be investigated, fixed, rerun, and verified. Before completion confirm every TODO was executed and verified, tests and environment checker ran, the final tree and scope were inspected, and no unfinished or blocked entries remain.
