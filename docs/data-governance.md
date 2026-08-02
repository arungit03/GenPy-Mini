# Data Governance (Planned)

**Status: no data has been collected. This document defines requirements for
Phase 2 and beyond; it does not describe anything implemented today.**

## Core principle

Public visibility is not the same as a license to train on. **Never assume
that publicly readable code is automatically unrestricted for model
training.** Every source used must have its license terms checked and
recorded before inclusion.

## Requirements for future dataset work

- **Licensing**: use only datasets and repositories whose license terms
  permit the intended use (training, redistribution of derived artifacts,
  etc.). Reject sources with unclear or incompatible licenses.
- **Provenance metadata**: preserve the source repository, commit/version,
  and license for every retained file, so any file can be traced back to its
  origin.
- **Attribution and redistribution**: respect attribution requirements and
  any redistribution restrictions the source license imposes.
- **Secret and PII detection**: scan for credentials, API keys, tokens, and
  personal data before a source enters the cleaned dataset, and remove or
  reject matches.
- **Generated and vendored code**: exclude auto-generated files and vendored
  third-party code that would otherwise duplicate or misattribute content.
- **Deduplication**: remove exact and near-duplicate files/snippets before
  training, both within and across sources.
- **Repository-level splitting**: split train/validation/test sets by
  repository (not by file or line) so related files can't leak across
  splits.
- **Train/test leakage prevention**: verify programmatically, not just by
  convention, that no evaluation content appears in the training split.
- **Dataset reports**: every dataset build must produce a report (source
  counts, license breakdown, dedup stats, split sizes) stored under
  `data/reports/`.

## What Phase 1 does not do

Phase 1 creates the empty, gitignored `data/` directory structure
(`raw/`, `interim/`, `cleaned/`, `tokenized/`, `instructions/`,
`evaluation/`, `reports/`) so later phases have a stable place to write to.
No data is collected, downloaded, cleaned, or committed in this phase, and
no dataset contents are ever committed to the repository — see
`.gitignore`.
