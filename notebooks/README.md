# Notebooks

No notebooks exist yet. This directory is reserved for future Kaggle
training notebooks and local exploratory notebooks (dataset inspection,
tokenizer inspection, evaluation review, etc.), introduced starting in
Phase 2 of [docs/roadmap.md](../docs/roadmap.md).

## Conventions (for future notebooks)

- Prefix filenames with the phase they belong to, e.g.
  `phase06-genpy-nano-smoke-test.ipynb`.
- Keep notebook outputs cleared before committing where practical; Jupyter
  checkpoint directories (`.ipynb_checkpoints/`) are gitignored.
- Import project code from the installed `genpy` package rather than
  duplicating logic inline, so notebook and library behaviour stay in sync.
- Kaggle-specific notebooks should not hard-code credentials — use Kaggle's
  own secrets mechanism, never a committed `.env` or config file.
