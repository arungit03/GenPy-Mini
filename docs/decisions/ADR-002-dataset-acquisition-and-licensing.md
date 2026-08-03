# ADR-002: Dataset Acquisition and Licensing

## Status

Accepted — 2026-08-03

## Context

Phase 2 needed to answer: how does GenPy-Mini bring external Python source
material into the project without repeating the mistake [ADR-001](ADR-001-project-scope.md)
and [data-governance.md](../data-governance.md) both warn against —
assuming public visibility equals a license to train? The system also has
to work within real constraints: a 16 GB Windows laptop with no GPU for
development, later training on ephemeral Kaggle sessions with limited
storage, and a strict rule against adding heavy dependencies before
they're needed.

## Decisions

### Acquisition is separate from cleaning

Phase 2 (`src/genpy/data/`) only registers, validates, and reproducibly
fetches raw source material with full provenance. It deliberately does
**not** parse Python ASTs, scan for secrets or personal data, deduplicate,
score quality, or split data into train/test sets — those are Phase 3.
Mixing acquisition and cleaning into one system would make failures harder
to diagnose (was it a license problem, a network problem, or a bad
regex?) and would make it impossible to re-run cleaning logic against
already-acquired raw data without re-downloading everything.

### Unknown licenses are blocked or require review, never auto-approved

`config/license_policy.yaml` classifies SPDX identifiers into `allowed`,
`review_required`, or `blocked`, with a configurable `default_status`
(currently `review_required`) for anything not explicitly listed. An
identifier is never treated as safe just because it's absent from the
`blocked` list. Combined with governance review
(`config/dataset_sources.yaml` → `governance.approval_status`), a source is
only ever `approved` when **both** the human review and the license policy
agree — see `src/genpy/data/source_registry.py::evaluate_source`. Either
one saying no is final and unoverridable; a `review_required` verdict can
only be bypassed with an explicit, recorded CLI override
(`--allow-review-required` + `--override-reason`).

### Sources require immutable revisions

A `git_repository` source pinned to a floating ref like `main` would make
every acquisition non-reproducible and would make "what code did we
actually train on" unanswerable months later. `defaults.require_pinned_revision`
rejects floating refs at registry-load time, and the acquisition system
records the fully resolved commit hash in the manifest regardless of what
ref was requested.

### JSON manifests, not a database

Every acquired source gets one deterministic, sorted, UTF-8 JSON file at
`data/manifests/<source-id>-<revision>.json`
(`src/genpy/data/manifests.py`). JSON was chosen over a database because
it needs zero new infrastructure, diffs cleanly in version control tooling
(even though the manifests themselves aren't committed by default), and is
trivially readable by both humans and the Phase 3+ pipeline without adding
a dependency.

### Raw data is excluded from Git

`data/raw/`, `data/manifests/`, and generated reports are all gitignored
(with `.gitkeep` placeholders preserved). Dataset content — however small
— does not belong in a source-controlled repository: licenses vary
per-file, sizes can grow unpredictably, and a `git log` should never
become an accidental redistribution channel for someone else's code.

### Acquisition is resource-bounded

Every source carries `maximum_download_bytes` and `maximum_extracted_bytes`,
enforced while streaming (not after the fact), because the target
development machine has ~300 GB free and 16 GB RAM, and Kaggle sessions
have limited persistent storage. Archive extraction also defends against
zip-slip, tar path traversal, and symlink/hardlink entries, since
untrusted archives are executed nowhere but their contents can still
overwrite arbitrary files on disk if extraction isn't careful.

### External APIs and heavy data libraries are deferred

No Hugging Face `datasets`, Kaggle API client, GitHub API client, or HTTP
client library beyond the standard library's `urllib` was added in Phase
2. Each of those is a real, justified dependency for a *later* phase (bulk
dataset ingestion, richer repository search) but none is required to
implement three source types (local directory, git, plain HTTPS archive)
safely. Keeping Phase 2 on the standard library plus the existing PyYAML
dependency keeps the dependency surface — and therefore the supply-chain
risk and install footprint on a 16 GB laptop — minimal until a future
phase's requirements actually justify more.

## Consequences

- Every future dataset source, however it's obtained, must go through this
  same registry/governance/manifest path — there is no "quick" bypass.
- Phase 3's cleaning pipeline can assume every input file under
  `data/raw/sources/` has a corresponding manifest recording its license
  and provenance, without re-deriving that information itself.
- Reports generated from manifests (`scripts/generate_acquisition_report.py`)
  must never claim acquired data is safe, clean, or training-ready — see
  the disclaimer required in every report.
- If a future dataset genuinely requires the Hugging Face `datasets`
  library or a provider-specific API, that should be its own ADR
  documenting why the standard-library approach is no longer sufficient.
