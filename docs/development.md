# Development Guide

GenPy-Mini targets **Python 3.11+**, a `src/` package layout, and works on
both Windows (local development) and Linux (CI and Kaggle). `make` is not
assumed to be installed on Windows, so every `make` target below has a
direct PowerShell equivalent.

## Windows (PowerShell) setup

```powershell
# From the repository root
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install
```

If `py -3.11` is not available, substitute whatever Python 3.11+ interpreter
is installed (e.g. `python -m venv .venv`) — check with `py -0p` or
`python --version` first.

## Linux / Kaggle setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install
```

Kaggle notebooks generally provide Python and pip directly; a virtual
environment is optional there but recommended for reproducibility when
scripting outside the notebook's own kernel.

## Quality commands

| Purpose | `make` target | PowerShell equivalent |
| --- | --- | --- |
| Install project + dev deps | `make install` | `pip install -e ".[dev]"` |
| Format code | `make format` | `ruff format .` |
| Lint | `make lint` | `ruff check .` |
| Type-check | `make typecheck` | `mypy` |
| Run tests | `make test` | `pytest` |
| Validate environment | `make validate` | `python scripts\validate_environment.py` |
| Format + lint + typecheck + test | `make quality` | run the four commands above in sequence |

Additional useful commands (no `make` target, run directly on either OS):

```bash
ruff format --check .        # verify formatting without changing files (what CI runs)
mypy                          # type-check src/, config/, scripts/, tests/
pytest --cov-report=term-missing   # tests with a coverage summary
pre-commit run --all-files    # run every configured pre-commit hook once
```

## Editable install notes

`pip install -e ".[dev]"` installs the `genpy` package from `src/genpy` in
editable mode. The top-level `config/` and `scripts/` directories are not
packaged or pip-installed — they are project-local Python packages used
directly from the repository root. `pytest` and `scripts/validate_environment.py`
both work without any extra `PYTHONPATH` setup: pytest resolves the repo
root automatically because none of `config/`, `scripts/`, or `tests/` has a
parent `__init__.py`, and `scripts/validate_environment.py` bootstraps its
own `sys.path` at the top of the file so it also runs standalone via
`python scripts/validate_environment.py`.

## No CUDA locally

The local machine has no CUDA-capable GPU. Nothing in Phase 1 (or the
`install`/`quality`/`validate` commands above) requires CUDA, PyTorch, or
any GPU driver. GPU-dependent work is deferred to the Kaggle training
phases in [docs/roadmap.md](roadmap.md).
