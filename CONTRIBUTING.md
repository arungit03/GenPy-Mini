# Contributing to GenPy-Mini

GenPy-Mini is currently in **Phase 1 (foundation)**. There is no dataset,
tokenizer, model, or training code yet — see [docs/roadmap.md](docs/roadmap.md)
for what each phase adds.

## Development setup

See [docs/development.md](docs/development.md) for full Windows PowerShell
and Linux/Kaggle setup instructions. In short:

```bash
python -m venv .venv
# activate the venv, then:
pip install -e ".[dev]"
pre-commit install
```

## Before opening a pull request

Run the full quality suite and make sure it passes:

```bash
ruff format .
ruff check .
mypy
pytest
python scripts/validate_environment.py
```

On Linux/macOS you can run `make quality` to do the first four in one step.

## Code standards

- Python 3.11+ syntax, `src/` package layout, full type hints everywhere.
- Use `pathlib.Path`, not string concatenation, for filesystem paths.
- No hard-coded absolute paths and no machine-specific usernames or directories.
- No secrets committed anywhere, including in tests or example data.
- Prefer the standard library. Only add a dependency when it clearly earns
  its place — see [docs/architecture.md](docs/architecture.md) for what is
  in scope for the current phase.
- Keep pull requests scoped to a single phase or concern where possible.

## Commit messages

Use short, imperative-mood summaries (e.g. `fix: correct dropout validation
bounds`). Conventional prefixes (`feat`, `fix`, `chore`, `docs`, `test`,
`refactor`) are welcome but not mandatory.

## Reporting issues

Since this is currently a single-maintainer, portfolio-scale project, open
an issue in the repository's issue tracker describing the problem, the
expected behaviour, and the phase it relates to.
