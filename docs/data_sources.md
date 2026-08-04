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

## Production Expansion Audit

The 2026-08-04 metadata-only audit did not approve or download new corpus content. It resolved
immutable revisions and inspected complete, untruncated Git trees for the following
permissively licensed candidates. `Eligible bytes` is the sum of 64-byte to 1-MiB Python blobs
after path-part exclusions; it is a planning upper bound before licence-conflict checks,
normalization, safety scanning, quality filtering, and deduplication.

| Candidate | Immutable revision | Root licence | Eligible files | Eligible bytes |
| --- | --- | --- | ---: | ---: |
| exercism/python | `250bf19050543da09e122211fc7430fe6f44c564` | MIT | 427 | 1,170,546 |
| pandas-dev/pandas | `30538196adabff39cfb2d809eb6119a96455b450` | BSD-3-Clause | 1,399 | 23,417,409 |
| scipy/scipy | `3112deba0d3f2ec816fa43d1faffc9754e722cd3` | BSD-3-Clause | 1,052 | 21,913,963 |
| scikit-learn/scikit-learn | `1a970e8b191106eac9e580192779dccc1b0177f6` | BSD-3-Clause | 953 | 15,793,724 |
| pytest-dev/pytest | `d81152a8376372c142ea1fcd464b060e52974cd5` | MIT | 260 | 3,732,217 |
| pydantic/pydantic | `b75fadbaa55d4d700a388f1b67683ef0be0ed540` | MIT | 394 | 6,066,277 |
| fastapi/fastapi | `b101622ec92844fe90a96f49c85accd8fada24df` | MIT | 946 | 3,968,976 |
| sqlalchemy/sqlalchemy | `2b780d9b9e78fe256a6968ea26980d051be2582b` | MIT | 646 | 20,467,286 |
| scrapy/scrapy | `298c9e610ec29dea378a706e925ad0d897007410` | BSD-3-Clause | 454 | 2,945,315 |
| aio-libs/aiohttp | `d5d068cb541ab7df5ecca14515475f9d4a379c5e` | Apache-2.0 | 163 | 3,220,246 |
| home-assistant/core | `9f74948e26a653888489eacae14e3fa338fa318e` | Apache-2.0 | 16,981 | 115,038,027 |
| apache/airflow | `b316afb44dc9686f5c39aab678a9a89cb21d53d2` | Apache-2.0 | 7,678 | 62,923,330 |
| pytorch/pytorch | `d47daf6e85dcd5a7ca79ac7b2ca73b1cb113cbd1` | BSD-3-Clause | 4,526 | 101,824,978 |
| tensorflow/tensorflow | `e30ad0b36aaae6d9b7b5a98e811b2e1356f6a117` | Apache-2.0 | 3,035 | 46,050,452 |
| ray-project/ray | `f8aa4d14edf4d58a10e94e86bd31920d1346ca6e` | Apache-2.0 | 4,382 | 46,797,683 |
| saltstack/salt | `c3e5bb1981a025a69bf94269282838ab618079c8` | Apache-2.0 | 2,680 | 33,307,247 |
| huggingface/transformers | `d09f53a801f45ad73ec3510e17972024234bc0fd` | Apache-2.0 | 4,165 | 74,502,970 |
| apache/superset | `e4ef84ca724b6a4de2e916fbbca04a989cbf5d21` | Apache-2.0 | 2,483 | 22,238,823 |
| networkx/networkx | `8751d1cfaacd8ca60f0a648abe4ecc13e5dc6b0c` | BSD-3-Clause | 663 | 6,849,215 |
| nltk/nltk | `35813c85af3f13d4f7196085854eafb3d0c5db02` | Apache-2.0 | 433 | 5,514,141 |

These candidates total 617,742,825 eligible raw Python bytes. Click retained 86.88% of its
eligible bytes after the current filters, which would project to 536,710,965 bytes, but that
single-repository ratio is not a readiness result. A controlled production build should first
inventory at least 805,306,368 eligible bytes (768 MiB) to retain useful headroom.

No candidate is approved yet: each pinned archive still needs an exact byte count and SHA-256,
nested-licence and vendored-path review, and a bounded dry run. The current 25-MiB per-source
download and 500-record limits prohibit that production run. NumPy, SymPy, and Celery remain
excluded because GitHub reports `NOASSERTION`; Django, Apache Beam, and Dagster require
nested-licence resolution; CPython is outside the allowlist; TheAlgorithms/Python is reserved
from training as algorithm/evaluation-like material; and The Stack v2 remains gated and
mixed-licence.

Exercism's pinned Python track is the only identified instruction candidate. Its tree contains
161 instruction Markdown files and 140 example Python files, but no records are approved: the
project still needs a provenance-complete deterministic pairing adapter, problem-family split
rules, archive checksum verification, and an explicit contamination review before use.
