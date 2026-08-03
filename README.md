# GenPy-Mini

A Python-only, decoder-only Transformer language model, trained from scratch.

## Current status: Phase 2 — Dataset Acquisition and Governance

Phase 1 (project structure, configuration system, quality infrastructure)
is complete. Phase 2 adds a governed, reproducible system for
**registering, reviewing, and acquiring raw Python source material** —
see [docs/dataset-acquisition.md](docs/dataset-acquisition.md).

**The dataset source registry (`config/dataset_sources.yaml`) ships empty.
No external dataset is acquired, enabled, or committed as part of this
repository. No tokenizer exists, no model has been implemented, and no
training has occurred. There is no trained or usable model in this
repository.**

### Dataset acquisition overview

`src/genpy/data/` implements three source types —
`local_directory`, `git_repository`, `http_archive` — each acquired
reproducibly (streamed, byte-limited, checksum-verified where applicable)
into `data/raw/sources/<id>/<revision>/`, with a full JSON provenance
manifest written to `data/manifests/`. This produces **raw, traceable
source material only**:

> Acquired data remains raw and has not yet passed cleaning, secret
> scanning, deduplication, quality filtering or train/test leakage
> controls.

### Governance-first approach

A source can only be acquired when **all** of the following are true:

1. It is registered in `config/dataset_sources.yaml` with `enabled: true`.
2. A human governance review recorded `approval_status: approved`.
3. Its declared SPDX license evaluates to `allowed` under
   `config/license_policy.yaml`.

**Registering a source does not mean it is legally approved.** A source
absent from the registry — or present but not approved — is refused by
default. `review_required` sources can only be acquired with an explicit,
recorded `--allow-review-required` + `--override-reason` override;
`rejected` sources can never be acquired. See
[docs/data-governance.md](docs/data-governance.md) and
[ADR-002](docs/decisions/ADR-002-dataset-acquisition-and-licensing.md).

### Dataset commands

```powershell
# Validate the registry + license policy without acquiring anything
python scripts/validate_sources.py --all

# See what would be acquired, without writing any files
python scripts/acquire_sources.py --all-approved --dry-run

# Acquire everything currently approved
python scripts/acquire_sources.py --all-approved

# Acquire one specific source
python scripts/acquire_sources.py --source <source-id>

# Regenerate the audit report from whatever has been acquired
python scripts/generate_acquisition_report.py `
  --manifest-dir data/manifests `
  --output-json data/reports/acquisition-report.json `
  --output-markdown data/reports/acquisition-report.md
```

### Storage considerations (300 GB free disk)

Every source has configurable `maximum_download_bytes` and
`maximum_extracted_bytes` limits, enforced while streaming — not after the
fact. On a machine with ~300 GB free, keep per-source limits modest (low
hundreds of MB) unless you've confirmed there's room for something larger;
`scripts/acquire_sources.py` prints the limits that will apply before
acquiring each source.

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
├── config/                         # model_config.yaml, dataset_sources.yaml,
│                                    # license_policy.yaml, settings.py + loaders
├── data/                           # empty, gitignored dataset/manifest/report directories
├── docs/                           # architecture, roadmap, governance, ADRs
├── notebooks/                      # reserved for future Kaggle/local notebooks
├── scripts/                        # validate_environment, validate_sources,
│                                    # acquire_sources, generate_acquisition_report
├── src/genpy/                      # installable package
│   ├── config.py, constants.py     # model configuration (Phase 1)
│   └── data/                       # dataset acquisition + governance (Phase 2)
├── tests/                          # config, repo-structure, and data/ tests
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

Phase 1 through Phase 17 (deployment and model card) are defined in
[docs/roadmap.md](docs/roadmap.md), each with its own deliverable and
acceptance gate. Phases 1 and 2 (foundation; dataset acquisition and
governance) are complete — tokenizer, model, training, inference, API,
frontend, and deployment are all future phases.

## Dataset licensing warning

**Registering a source in `config/dataset_sources.yaml` does not make it
legally approved.** Every source requires an explicit human governance
review *and* a license that evaluates to `allowed` under
`config/license_policy.yaml` before it can be acquired — enforced by
`src/genpy/data/`, not just documented. Public visibility of code is
**not** the same as a license to train on it. See
[docs/data-governance.md](docs/data-governance.md) and
[docs/dataset-acquisition.md](docs/dataset-acquisition.md).

## Security warning: generated code execution

No inference or code-execution path exists yet. When it does: **model
-generated code must never be executed outside an isolated sandbox** (no
network access, no host filesystem access, resource-limited). Generated
code is untrusted output, not trusted input, and must be treated that way
by any tooling that consumes it. See Phase 15 of
[docs/roadmap.md](docs/roadmap.md).

## Known current limitations

- The dataset source registry ships empty; no external dataset is
  acquired, enabled, or committed. No tokenizer, model, training,
  inference, API, or frontend code exists.
- Phase 2 acquires raw source material only — no secret/PII scanning,
  deduplication, quality filtering, or train/test-safe splitting exists
  yet (Phase 3).
- `git_repository` acquisition does not enforce `maximum_download_bytes`
  during the clone/fetch itself (only the resulting working-tree size via
  `maximum_extracted_bytes`); see
  [docs/dataset-acquisition.md](docs/dataset-acquisition.md).
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
