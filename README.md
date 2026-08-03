# GenPy

GenPy is a small Python-focused code-generation language model. Its goal is to convert
natural-language programming requests into readable beginner-friendly Python code.

Example target behavior:

```text
Write a program to check odd or even
```

The model should return only the Python program unless the user explicitly asks for an
explanation.

## Project Status

Current phase: **Phase 2 - Dataset preparation**.

Phase 1 is complete. The Phase 2 pipeline now provides source and licence auditing,
bounded resumable ingestion, deterministic cleaning, static secret/PII/safety filtering,
Python validation, disk-backed deduplication, group-aware splits, Zstandard JSONL shards,
and provenance/statistics reports. A bounded real-source smoke run is verified; the
50-100M-token GenPy-5M corpus has not been approved or built.

## Important Constraints

- GenPy will be trained from completely random initial weights.
- No pretrained LLM weights, adapters, or tokenizers are allowed.
- The tokenizer will be a custom byte-level BPE tokenizer trained only on GenPy data.
- The final tokenizer vocabulary target is 16,384 tokens.
- The final context length target is 1,024 tokens.
- The final model target is approximately 100 million parameters.
- Training is planned for Kaggle GPU.
- Local development, testing, and inference must work on Windows CPU-only systems.

## Main Goals

- Generate simple and intermediate Python programs from natural-language prompts.
- Keep outputs readable and beginner-friendly.
- Avoid unnecessary libraries for basic tasks.
- Evaluate model quality using syntax checks, execution tests, output-format checks, and
  safety checks, not training loss alone.
- Scale carefully from GenPy-5M to GenPy-25M to GenPy-100M.

## Planned Architecture

GenPy is planned as a decoder-only Transformer using:

- Causal self-attention
- RoPE positional encoding
- RMSNorm
- SwiGLU feed-forward layers
- Tied input/output embeddings
- 1,024-token context length
- 16,384-token vocabulary

These choices are standard and practical for a small code-generation model, but they do
not guarantee strong model quality.

## Repository Structure

```text
GenPy/
|-- AGENTS.md
|-- README.md
|-- LICENSE
|-- pyproject.toml
|-- requirements.txt
|-- requirements-dev.txt
|-- requirements-kaggle.txt
|-- configs/
|   |-- model/
|   |   |-- genpy_5m.yaml
|   |   |-- genpy_25m.yaml
|   |   `-- genpy_100m.yaml
|   |-- pretrain.yaml
|   |-- instruction_train.yaml
|   `-- evaluate.yaml
|-- data/
|-- docs/
|-- notebooks/
|-- scripts/
|-- src/genpy/
|-- tests/
|-- artifacts/
|-- checkpoints/
`-- logs/
```

## Windows PowerShell Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-data.txt
python scripts/check_environment.py
pytest
```

## Linux or Kaggle Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-data.txt
python scripts/check_environment.py
pytest
```

For Kaggle training notebooks, install `requirements-kaggle.txt` instead of the local
development dependency set when appropriate. GPU availability must be checked through
PyTorch with `torch.cuda.is_available()`.

## Useful Commands

Verify the environment:

```bash
python scripts/check_environment.py
```

Run tests:

```bash
pytest
```

Run lint:

```bash
ruff check .
```

## Dataset Policy

Only records with adequate provenance and a licence on the provisional allowlist may
enter the corpus. Missing, custom, copyleft, non-commercial, research-only, and conflicting
licences are excluded by default. This policy is conservative project policy, not legal
advice. Repository/file attribution and validated opt-out requests must be retained.

Dataset contents, quarantine material, caches, shards, reports containing local metadata,
and download state are ignored by Git. Large builds require a disk review and explicit
confirmation. Do not add raw or processed dataset files to commits.

Audit and run the bounded pipeline:

```powershell
python scripts/data/audit_sources.py --config configs/data/sources.yaml
python scripts/data/build_dataset.py --config configs/data/phase2.yaml --mode smoke
python scripts/data/validate_dataset.py --config configs/data/phase2.yaml
python scripts/data/create_splits.py --config configs/data/phase2.yaml
python scripts/data/generate_dataset_report.py --config configs/data/phase2.yaml
```

The exact next phase is Phase 3: train and validate the custom 16,384-token byte-level BPE
tokenizer only after a corpus is approved and frozen. No pretrained tokenizer may be used.

## Roadmap

1. Planning and setup
2. Dataset preparation
3. Custom tokenizer
4. Model implementation
5. Training from random weights
6. Evaluation and improvement
7. Application and release

## Known Limitations

- GenPy V1 will not compete with large commercial LLMs.
- It may fail on advanced algorithms and larger projects.
- It will not initially generate full-stack applications.
- It will not initially target advanced frameworks such as Django, PyTorch, or TensorFlow.
- It will not guarantee that every generated program is correct.

## Safety Note

Generated code must not be executed directly on a user's main operating system. Any future
execution feature must use an isolated sandbox with strict network, time, memory, and
filesystem limits.
