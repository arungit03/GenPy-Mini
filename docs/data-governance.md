# Data Governance

**Status: Phase 2 (source acquisition and governance) is implemented and
enforced by tooling — see `src/genpy/data/` and
[dataset-acquisition.md](dataset-acquisition.md). Phase 3 (cleaning, secret
and PII scanning, deduplication, quality filtering, and train/test-safe
splitting) is not yet implemented; the requirements below for those areas
remain planned.**

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

## Phase 2 rules (implemented and enforced)

These rules are enforced by `src/genpy/data/` and the three scripts in
`scripts/` — not just documented here. See
[dataset-acquisition.md](dataset-acquisition.md) for the full workflow.

- **Source approval**: a source may only be acquired when it is
  `enabled: true` in `config/dataset_sources.yaml` *and*
  `src/genpy/data/source_registry.py::evaluate_source()` returns
  `approved` — which itself requires both a human governance sign-off
  (`governance.approval_status: approved`) and an `allowed` license
  determination. A source absent from the registry, or present but not
  approved, is refused by default.
- **License policy**: `config/license_policy.yaml` classifies SPDX
  identifiers as `allowed`, `review_required`, or `blocked`, with a
  configurable fallback for anything unlisted (`review_required` by
  default — never silently `allowed`). It is explicitly not legal advice;
  see its own `disclaimer` field and [ADR-002](decisions/ADR-002-dataset-acquisition-and-licensing.md).
- **Provenance retention**: every acquired source gets a JSON manifest
  (`data/manifests/<source-id>-<revision>.json`) recording its resolved
  revision, declared license, governance status, and a per-file SHA-256 —
  see `src/genpy/data/manifests.py`.
- **Immutable revisions**: `git_repository` sources must use a pinned
  commit hash or tag while `defaults.require_pinned_revision` is `true`;
  floating refs (`main`, `HEAD`, etc.) are rejected at registry-load time.
- **Attribution records**: `license.attribution_required` is carried
  through into every manifest and surfaced in the acquisition report's
  "Attribution required" section, so nothing acquired can silently lose
  its attribution obligation.
- **Overrides**: a `rejected` source can never be acquired. A
  `review_required` source can only be acquired with an explicit
  `--allow-review-required` flag and a non-empty `--override-reason`,
  both of which are recorded in the manifest and the acquisition report.
- **Raw-data status**: acquired data is raw. Every generated report
  states, verbatim: *"Acquired data remains raw and has not yet passed
  cleaning, secret scanning, deduplication, quality filtering or
  train/test leakage controls."*
- **Audit reports**: `scripts/generate_acquisition_report.py` produces
  both `data/reports/acquisition-report.json` and
  `data/reports/acquisition-report.md`, distinguishing registered,
  enabled, approved, acquired, review-required, rejected, and
  failed/incomplete sources, plus total raw storage consumed.

## What is not yet implemented (Phase 3+)

Phase 2 acquires raw, traceable source material only. It does **not**:

- Detect secrets, API keys, tokens, or personal data in acquired content.
- Exclude generated or vendored code.
- Deduplicate files or snippets, exactly or approximately.
- Split data by repository into train/validation/test sets.
- Verify the absence of train/test leakage.
- Produce a dataset ready for tokenization or training.

All of the above remain Phase 3 requirements (see
[roadmap.md](roadmap.md)) and are not satisfied merely by running
`scripts/acquire_sources.py`.

`data/raw/`, `data/manifests/`, and generated reports are all gitignored
(with `.gitkeep` placeholders preserved) — no dataset contents, provenance
manifests, or acquisition reports are ever committed to the repository.
See `.gitignore`.
