# GenPy-Mini

A Python-only, decoder-only Transformer language model, trained from scratch.

## Current status: Phase 1 — Foundation

This repository currently contains **only** the Phase 1 foundation: project
structure, documented target architecture, a validated configuration
system, development standards, and quality/validation infrastructure.

**No dataset has been collected, no tokenizer exists, no model has been
implemented, and no training has occurred. There is no trained or usable
model in this repository.**

## Objective

GenPy-Mini is an educational, portfolio-scale project to build a working
large language model system end-to-end, from scratch, for Python:

- Natural language → Python code generation
- Python code completion
- Python debugging assistance
- Python code explanation
- Python code optimisation
- Python unit-test generation

It is explicitly **not** intended to match commercial coding assistants —
see [ADR-001](docs/decisions/ADR-001-project-scope.md) for the reasoning
behind the project's scope.

## Target architecture

Stored, validated configuration data — see
[`config/model_config.yaml`](config/model_config.yaml) and
[`src/genpy/config.py`](src/genpy/config.py). Not yet consumed by any model
code.

| Parameter | Value |
| --- | --- |
| Architecture | Decoder-only Transformer |
| Task | Causal language modeling |
| Language scope | Python, English programming instructions |
| Vocabulary size | 16,000 |
| Context length | 512 tokens |
| Model dimension (`d_model`) | 512 |
| Layers | 8 |
| Attention heads | 8 |
| Feed-forward dimension | 2,048 |
| Dropout | 0.1 |
| Tied input/output embeddings | Yes |
| Target parameter count | 33,000,000 – 35,000,000 |

See [docs/architecture.md](docs/architecture.md) for the full planned data
→ model → serving pipeline.

## Local vs. Kaggle responsibilities

| Responsibility | Local machine | Kaggle |
| --- | --- | --- |
| Coding, docs, configuration | Yes | No |
| Unit tests / CPU smoke tests | Yes | No |
| Dataset preprocessing (future) | Yes | No |
| GPU pretraining / fine-tuning (future) | No | Yes |

The local development machine (Windows 11, Intel Core i7-13620H, 16 GB RAM,
integrated GPU) never requires CUDA. GPU training happens on Kaggle
notebooks in a later phase.

## Repository structure

```text
GenPy-Mini/
├── .github/workflows/quality.yml   # CI: ruff, mypy, pytest
├── config/                         # model_config.yaml + settings.py + loader
├── data/                           # empty, gitignored dataset directories
├── docs/                           # architecture, roadmap, governance, ADRs
├── notebooks/                      # reserved for future Kaggle/local notebooks
├── scripts/validate_environment.py # local environment/repo health check
├── src/genpy/                      # installable package (config loader, constants)
├── tests/                          # config + repo-structure tests
├── pyproject.toml
└── requirements-dev.txt
```

## Installation — Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install
```

## Installation — Linux / Kaggle

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install
```

See [docs/development.md](docs/development.md) for full details, including
a PowerShell equivalent for every `make` target.

## Quality-check commands

```bash
ruff format .          # or: make format
ruff check .           # or: make lint
mypy                   # or: make typecheck
pytest                 # or: make test
python scripts/validate_environment.py   # or: make validate
```

## Development roadmap

Phase 1 (this repository, today) through Phase 17 (deployment and model
card) are defined in [docs/roadmap.md](docs/roadmap.md), each with its own
deliverable and acceptance gate. Phase 1 covers structure and configuration
only — dataset, tokenizer, model, training, inference, API, frontend, and
deployment are all future phases.

## Dataset licensing warning

No dataset has been collected yet. When one is, every source's license will
be checked and recorded before inclusion — public visibility of code is
**not** the same as a license to train on it. See
[docs/data-governance.md](docs/data-governance.md).

## Security warning: generated code execution

No inference or code-execution path exists yet. When it does: **model
-generated code must never be executed outside an isolated sandbox** (no
network access, no host filesystem access, resource-limited). Generated
code is untrusted output, not trusted input, and must be treated that way
by any tooling that consumes it. See Phase 15 of
[docs/roadmap.md](docs/roadmap.md).

## Known current limitations

- No dataset, tokenizer, model, training, inference, API, or frontend code
  exists.
- The architecture in `config/model_config.yaml` is a target specification
  only; nothing has verified the actual parameter count of a real model
  built from it, because no model exists yet.
- Local development has no CUDA-capable GPU; GPU-dependent work is entirely
  deferred to Kaggle.
- No project license has been selected yet — see
  [LICENSE-NOTICE.md](LICENSE-NOTICE.md).

## No trained model exists

To be unambiguous: **this repository does not contain, and cannot yet
produce, a trained or usable GenPy-Mini model.** Nothing in this README or
elsewhere in the repository should be read as a performance claim.
