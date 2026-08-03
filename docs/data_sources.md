# Data Sources

Source decisions are pinned in `configs/data/sources.yaml`. Unknown values stay unknown;
entries are not approved merely because they are listed.

| Source | Revision | Licence evidence | Access and scale | Decision |
| --- | --- | --- | --- | --- |
| [Click](https://github.com/pallets/click) | `934813e4d421071a1b3db3973c02fe2721359a6e` (tag 8.1.8) | [BSD-3-Clause](https://github.com/pallets/click/blob/8.1.8/LICENSE.txt) | Public pinned archive; 399,834-byte verified download | Approved for bounded smoke sample |
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | `v2.1.0` | Dataset card says `other`; per-record detected licences | Gated/mixed source; 32.1 TB near-deduplicated and 67.5 TB full multilingual upstream | Review required; excluded |

Click is a single-repository verification source, not a proposed final corpus. Its archive is
pinned by commit and SHA-256. Repository-level BSD-3-Clause applies, and provenance remains
attached to every record.

The Stack v2 offers record provenance, detected licence metadata, sharded access, and an
opt-out process, but it includes mixed or missing licences and requires gated access and
upstream terms. GenPy will not request credentials or ingest it automatically. Approval would
still require a Python-only size estimate, per-record allowlist enforcement, current opt-out
handling, attribution planning, and adequate disk capacity.

No complete multi-gigabyte corpus download has been authorized. A planning estimate for an
eventual curated 1-2B-token corpus is roughly 3.5-7 GB of normalized text under the current
byte/token heuristic, but raw sources, indexes, manifests, quarantine, and build headroom can
raise working disk needs substantially. Reserve at least 50-100 GB only as an initial planning
range; measure approved source shards before committing to a full build.
