# Dataset Source Templates

Complete, copy-pasteable examples for each supported `source_type` in
`config/dataset_sources.yaml`. Every example below uses placeholder values
and is `enabled: false` — copy the block you need into the `sources:` list,
replace the placeholders, and leave it disabled until a human reviewer has
actually completed the governance review.

See [dataset-acquisition.md](dataset-acquisition.md) for the full workflow
and [data-governance.md](data-governance.md) for what a real review must
check.

## `local_directory`

For a dataset you already have on disk (e.g. a manually vetted snapshot).

```yaml
- id: my-local-python-snapshot
  name: My Local Python Snapshot
  enabled: false
  source_type: local_directory
  location: C:/datasets/my-python-snapshot   # absolute path; forward slashes work on Windows too
  description: >
    A manually curated local snapshot of Python files, already reviewed
    for licensing before being placed on disk.
  revision: 2026-08-03-manual-snapshot        # any stable label you choose
  license:
    declared_spdx: MIT
    license_file: LICENSE
    attribution_required: true
    redistribution_allowed: true
    commercial_use_allowed: true
    modifications_allowed: true
    notes: Reviewed against the upstream LICENSE file on 2026-08-03.
  governance:
    reviewed_by: Your Name
    reviewed_on: 2026-08-03
    approval_status: review_required
    approval_notes: Pending final sign-off.
  acquisition:
    expected_sha256: null
    maximum_download_bytes: 536870912   # 512 MiB
    maximum_extracted_bytes: 536870912
    include_submodules: false
    shallow_clone: true
  tags:
    - python
    - manual
```

## `git_repository`

For a specific, pinned commit or tag of a git repository. Floating refs
(`main`, `master`, `HEAD`, `latest`) are rejected while
`defaults.require_pinned_revision` is `true`.

```yaml
- id: example-git-source
  name: Example Git Source
  enabled: false
  source_type: git_repository
  location: https://example.invalid/some-org/some-repo.git
  description: A pinned snapshot of an example repository.
  revision: a1b2c3d4e5f60718293a4b5c6d7e8f9012345678   # full commit hash or an immutable tag
  license:
    declared_spdx: Apache-2.0
    license_file: LICENSE
    attribution_required: true
    redistribution_allowed: true
    commercial_use_allowed: true
    modifications_allowed: true
    notes: Verified against upstream LICENSE at the pinned commit.
  governance:
    reviewed_by: Your Name
    reviewed_on: 2026-08-03
    approval_status: review_required
    approval_notes: Pending final sign-off.
  acquisition:
    expected_sha256: null            # not applicable to git_repository
    maximum_download_bytes: 268435456   # 256 MiB (best-effort; see docs/dataset-acquisition.md)
    maximum_extracted_bytes: 268435456
    include_submodules: false        # submodules are not supported in Phase 2
    shallow_clone: true              # fetches only the pinned revision (depth 1)
  tags:
    - python
    - git
```

## `http_archive`

For a `.zip`, `.tar`, `.tar.gz`, or `.tgz` served over HTTPS. A SHA-256 is
required while `defaults.require_checksum_for_http_archives` is `true`.

```yaml
- id: example-archive-source
  name: Example Archive Source
  enabled: false
  source_type: http_archive
  location: https://example.invalid/datasets/example-archive-v1.tar.gz
  description: A versioned archive published by an example dataset provider.
  revision: v1.0.0
  license:
    declared_spdx: CC0-1.0
    license_file: LICENSE
    attribution_required: false
    redistribution_allowed: true
    commercial_use_allowed: true
    modifications_allowed: true
    notes: Public-domain dedication per the provider's dataset card.
  governance:
    reviewed_by: Your Name
    reviewed_on: 2026-08-03
    approval_status: review_required
    approval_notes: Pending final sign-off.
  acquisition:
    expected_sha256: 0000000000000000000000000000000000000000000000000000000000000000  # replace with the real digest
    maximum_download_bytes: 1073741824    # 1 GiB
    maximum_extracted_bytes: 2147483648   # 2 GiB
    include_submodules: false
    shallow_clone: true
  tags:
    - python
    - archive
  dataset_card: https://example.invalid/datasets/example-archive/card
  citation: "Example Provider (2026). Example Archive v1.0.0."
```

## Optional metadata fields

Any source may also set: `homepage`, `source_repository`, `dataset_card`,
`citation`, `contact`, `publication`, `known_restrictions`,
`attribution_text`. These are all plain strings and are validated the same
way as every other field — a typo'd field name is rejected, not silently
dropped.
