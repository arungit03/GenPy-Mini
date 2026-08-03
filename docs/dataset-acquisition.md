# Dataset Acquisition (Phase 2)

This document describes the governed acquisition system implemented in
`src/genpy/data/` and its three CLI entry points in `scripts/`. It produces
**raw, traceable source material only** — see the disclaimer repeated
throughout this document and in every generated report.

> Acquired data remains raw and has not yet passed cleaning, secret
> scanning, deduplication, quality filtering or train/test leakage
> controls.

Cleaning, secret/PII scanning, deduplication, quality scoring, and
repository-level splitting are Phase 3 (see [roadmap.md](roadmap.md)).

## Registry format

`config/dataset_sources.yaml` is the single source of truth for what may be
acquired. It starts empty on purpose. Its shape:

```yaml
schema_version: 1
defaults:
  enabled: false
  timeout_seconds: 60
  retry_count: 3
  maximum_download_bytes: 5368709120
  maximum_extracted_bytes: 10737418240
  require_pinned_revision: true
  require_license_metadata: true
  require_checksum_for_http_archives: true
sources: []
```

Every entry under `sources:` is validated by
`src/genpy/data/source_registry.py::load_source_registry()` against the
full schema in `src/genpy/data/schemas.py`. Unknown or misspelled fields
(at the registry, source, license, governance, or acquisition level) are
rejected outright rather than silently ignored. See
[dataset-source-template.md](dataset-source-template.md) for complete,
copy-pasteable examples of all three source types.

## Approval workflow

Registering a source is not the same as approving it, and approving it is
not the same as acquiring it. Three independent gates must all pass:

1. **`enabled: true`** in the registry entry.
2. **Governance approval** (`governance.approval_status: approved`) — a
   human reviewer's sign-off, recorded with reviewer name, date, and notes.
3. **License policy evaluation** — the declared SPDX identifier must
   evaluate to `allowed` under `config/license_policy.yaml`
   (`src/genpy/data/licenses.py::LicensePolicy.evaluate`).

`src/genpy/data/source_registry.py::evaluate_source()` combines governance
and license status conservatively: a source is `approved` only when *both*
say yes. Either one saying "rejected"/"blocked" makes the source
`rejected` outright (never overridable). Otherwise, any `review_required`
signal makes the whole source `review_required`.

**Public availability of code is never treated as a license grant.** An
SPDX identifier being on the `allowed` list means that license *family* is
broadly compatible with training use in common cases — it does not mean
every repository carrying that label has been individually cleared. See
the mandatory disclaimer in `config/license_policy.yaml`.

## Supported source types

| Type | What it does |
| --- | --- |
| `local_directory` | Stream-copies an existing local directory tree. |
| `git_repository` | Shallow-fetches a pinned revision via `git`, checks it out, copies the working tree (excluding `.git`). |
| `http_archive` | Streams a `.zip` / `.tar` / `.tar.gz` / `.tgz` over HTTPS, verifies its SHA-256, and safely extracts it. |

No Hugging Face, Kaggle API, or GitHub API integration exists in Phase 2 —
only local paths, `git`, and plain HTTPS downloads.

## Reproducibility requirements

- **Pinned revisions**: when `defaults.require_pinned_revision` is `true`
  (the default), `git_repository` sources must use an immutable commit
  hash or tag — floating refs like `main`, `master`, `HEAD`, or `latest`
  are rejected at registry-load time.
- **Hash verification**: `http_archive` sources require
  `acquisition.expected_sha256` when
  `defaults.require_checksum_for_http_archives` is `true` (the default).
  The downloaded file's SHA-256 is verified with a constant-time
  comparison before extraction ever begins.
- **Deterministic manifests**: every acquired source gets a JSON manifest
  at `data/manifests/<source-id>-<revision>.json` with a sorted file list
  and a `manifest_digest` summarizing the whole snapshot
  (`src/genpy/data/checksums.py::directory_digest`).

## Storage limits

Every source has `acquisition.maximum_download_bytes` and
`acquisition.maximum_extracted_bytes` (falling back to the registry
`defaults` when unset). `http_archive` downloads are streamed to disk and
aborted mid-transfer the instant the download limit would be exceeded;
extraction (zip/tar) and local/git copies are aborted the instant the
extracted-size limit would be exceeded. On the 16 GB local machine with
~300 GB free disk described in
[development.md](development.md#no-cuda-locally), keep per-source limits
modest (low hundreds of MB) unless you have verified free disk space for
larger sources.

`maximum_extracted_bytes` is the effective cap on final copied/extracted
content for **all three** source types (nothing is "downloaded" for
`local_directory` or measurable-in-advance for `git_repository`, so this
one limit governs the on-disk result in both cases).

## Manifest structure

See `src/genpy/data/manifests.py`. Every manifest is UTF-8 JSON with a
fixed key order (stable for diffs), forward-slash relative paths, and
never contains credentials — `configured_location` has any userinfo, query
string, and fragment stripped from URLs
(`src/genpy/data/manifests.py::redact_url_credentials`).

`scripts/generate_acquisition_report.py` verifies each manifest against the
acquired files on disk to compute "failed or incomplete" status. It assumes
the standard layout (`data/manifests/` and `data/raw/sources/` as siblings
under `data/`) by default; pass `--sources-root` explicitly if you point
`--manifest-dir` somewhere non-standard (e.g. a temporary directory used
for testing), or verification will look in the wrong place.

## Dry-run workflow

Always run with `--dry-run` first:

```powershell
python scripts/validate_sources.py --all
python scripts/acquire_sources.py --all-approved --dry-run
```

`--dry-run` never writes a file, directory, or manifest — it only reports
what *would* happen, including the byte limits that would apply.

## Override policy

- A `rejected` source can **never** be acquired, with or without any flag.
- A `review_required` source can only be acquired with **both**
  `--allow-review-required` and a non-empty `--override-reason`. The
  reason is recorded in that source's manifest
  (`acquisition.governance_override` / `acquisition.override_reason`) and
  surfaced in the acquisition report's "Governance overrides used"
  section.
- Using an override is not legal sign-off — it is a recorded, auditable
  exception, and the underlying governance review should still be
  completed and the registry updated to `approved` once it is.

## Failure recovery

Acquisition is idempotent and designed to fail safely:

- A failed acquisition never leaves a manifest-less "final" directory
  behind — see `src/genpy/data/acquisition.py::acquire_source`'s cleanup
  logic.
- A **successful** re-run against an already-acquired, manifest-verified
  source is a safe no-op (`status: "skipped"`). Use `--force` to
  reacquire.
- If `destination` and its manifest ever disagree (e.g. someone manually
  edited an acquired file), the next non-forced run refuses to proceed and
  asks for `--force` rather than silently trusting either side.
- Known limitation: the final "swap in the new content" and "write the new
  manifest" are two separate filesystem operations, not one atomic
  transaction. A crash in the narrow window between them could leave a
  manifest that doesn't match its directory — the *next* run's manifest
  verification will catch this and demand `--force` rather than silently
  trusting stale state.
- Partial HTTP downloads are written to a `*.partial` file (gitignored)
  and always removed on failure or after a checksum mismatch.

## Windows and Kaggle usage

Everything here works with the standard library plus the existing PyYAML
dependency — no new runtime dependency was added for Phase 2 (see
[README.md](../README.md) for the full quality-check commands). On
Windows, use the PowerShell examples above directly. On Kaggle (Linux),
the same scripts run unchanged with `python3 scripts/...`; remember Kaggle
sessions have limited persistent storage, so keep `maximum_download_bytes`
/ `maximum_extracted_bytes` conservative there too.

## Removing an acquired source safely

To remove one acquired source's raw data and its manifest:

1. Delete `data/raw/sources/<source-id>/<revision>/`.
2. Delete `data/manifests/<source-id>-<revision>.json`.
3. Regenerate the report:
   `python scripts/generate_acquisition_report.py --manifest-dir data/manifests --output-json data/reports/acquisition-report.json --output-markdown data/reports/acquisition-report.md`

Both paths are entirely within the gitignored `data/` tree, so removal
never touches version control.
