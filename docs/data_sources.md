# Data Sources

Source decisions are pinned in `configs/data/sources.yaml`. Unknown values stay unknown;
entries are not approved merely because they are listed.

| Source | Revision | Licence evidence | Access and scale | Decision |
| --- | --- | --- | --- | --- |
| [Click](https://github.com/pallets/click) | `934813e4d421071a1b3db3973c02fe2721359a6e` (tag 8.1.8) | [BSD-3-Clause](https://github.com/pallets/click/blob/8.1.8/LICENSE.txt) | Public pinned archive; 399,834-byte verified download | Approved; 63 records in corpus |
| [pandas](https://github.com/pandas-dev/pandas) | `30538196adabff39cfb2d809eb6119a96455b450` | [BSD-3-Clause](https://github.com/pandas-dev/pandas/blob/30538196adabff39cfb2d809eb6119a96455b450/LICENSE) | 6,030,910-byte verified download | Approved; 684 records in corpus |
| [SciPy](https://github.com/scipy/scipy) | `3112deba0d3f2ec816fa43d1faffc9754e722cd3` | [BSD-3-Clause](https://github.com/scipy/scipy/blob/3112deba0d3f2ec816fa43d1faffc9754e722cd3/LICENSE.txt) | 26,611,844-byte verified download | Approved with 7 nested-licence path exclusions; 582 records in corpus |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | `1a970e8b191106eac9e580192779dccc1b0177f6` | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/1a970e8b191106eac9e580192779dccc1b0177f6/COPYING) | 8,795,779-byte verified download | Approved with `sklearn/externals/*` excluded (vendored); 653 records in corpus |
| [pytest](https://github.com/pytest-dev/pytest) | `d81152a8376372c142ea1fcd464b060e52974cd5` | [MIT](https://github.com/pytest-dev/pytest/blob/d81152a8376372c142ea1fcd464b060e52974cd5/LICENSE) | 2,039,847-byte verified download | Approved; 216 records in corpus |
| [Pydantic](https://github.com/pydantic/pydantic) | `b75fadbaa55d4d700a388f1b67683ef0be0ed540` | [MIT](https://github.com/pydantic/pydantic/blob/b75fadbaa55d4d700a388f1b67683ef0be0ed540/LICENSE) | 3,773,377-byte verified download | Approved with `pydantic-core/*` excluded (separately licensed subpackage); 297 records in corpus |
| [FastAPI](https://github.com/fastapi/fastapi) | `b101622ec92844fe90a96f49c85accd8fada24df` | [MIT](https://github.com/fastapi/fastapi/blob/b101622ec92844fe90a96f49c85accd8fada24df/LICENSE) | 19,922,470-byte verified download | Approved; 770 records in corpus |
| [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) | `2b780d9b9e78fe256a6968ea26980d051be2582b` | [MIT](https://github.com/sqlalchemy/sqlalchemy/blob/2b780d9b9e78fe256a6968ea26980d051be2582b/LICENSE) | 5,960,400-byte verified download | Approved; 256 records in corpus (entire repository hashed into the validation split) |
| [Scrapy](https://github.com/scrapy/scrapy) | `298c9e610ec29dea378a706e925ad0d897007410` | [BSD-3-Clause](https://github.com/scrapy/scrapy/blob/298c9e610ec29dea378a706e925ad0d897007410/LICENSE) | 1,760,487-byte verified download | Approved; 390 records in corpus |
| [aiohttp](https://github.com/aio-libs/aiohttp) | `d5d068cb541ab7df5ecca14515475f9d4a379c5e` | [Apache-2.0](https://github.com/aio-libs/aiohttp/blob/d5d068cb541ab7df5ecca14515475f9d4a379c5e/LICENSE.txt) | 1,434,763-byte verified download | Approved; 99 records in corpus |
| [NetworkX](https://github.com/networkx/networkx) | `8751d1cfaacd8ca60f0a648abe4ecc13e5dc6b0c` | BSD-3-Clause (manually verified; see note) | 3,213,903-byte verified download | Approved; 454 records in corpus |
| [NLTK](https://github.com/nltk/nltk) | `35813c85af3f13d4f7196085854eafb3d0c5db02` | [Apache-2.0](https://github.com/nltk/nltk/blob/35813c85af3f13d4f7196085854eafb3d0c5db02/LICENSE.txt) | 3,404,717-byte verified download | Approved with 1 file excluded (non-commercial corpus reference); 96 records in corpus |
| [Exercism Python track](https://github.com/exercism/python) | `250bf19050543da09e122211fc7430fe6f44c564` | [MIT](https://github.com/exercism/python/blob/250bf19050543da09e122211fc7430fe6f44c564/LICENSE) | 2,634,797-byte verified download | Approved, instruction-only; 114 of 140 exercises accepted |
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | `v2.1.0` | Dataset card says `other`; per-record detected licences | Gated/mixed source; 32.1 TB near-deduplicated and 67.5 TB full multilingual upstream | Review required; excluded |

All twelve pinned pretraining sources above were downloaded via their pinned commit archive,
SHA-256 verified against `configs/data/sources.yaml`, extracted with path-traversal-safe
zip handling, and processed through the full Phase 2 pipeline (licence gate, UTF-8
normalization, secret/PII/unsafe-content scanning, Python 3.11 AST/tokenize validation, quality
scoring, exact and near deduplication, group-aware splitting) on 2026-08-04. Click is the
original single-repository verification source; the other eleven are the 2026-08-04 production
expansion. Repository-level licences apply per source, and provenance remains attached to every
record. NetworkX's GitHub-reported licence is `NOASSERTION` because its `LICENSE.txt` wraps the
3-clause BSD text in a reST literal block that the automated detector does not parse; the raw
file was read directly and manually confirmed as standard 3-clause BSD language.

## Nested-licence and vendoring review

Before approval, every candidate's extracted tree was scanned for nested `LICENSE`/`COPYING`/
`NOTICE` files below the repository root and for copyleft/non-commercial keywords in `.py`
file headers. Findings and the resulting `exclude_globs` in `configs/data/sources.yaml`:

- **SciPy**: 14 nested-licence subtrees found. Most (`optimize/tnc`, `optimize/_direct`,
  `sparse/linalg/_dsolve/SuperLU`, `sparse/linalg/_eigen/arpack/arnaud`,
  `sparse/linalg/_propack/PROPACK`, `stats/biasedurn`, `subprojects/qhull_r`,
  `subprojects/duccfft`) contain zero `.py` files and were never selectable anyway. Seven
  subtrees do contain `.py` files under a different copyright holder than SciPy's own
  BSD-3-Clause (`ndimage`, `fft/_duccfft`, `io/_fast_matrix_market`, `_lib/_uarray`,
  `_external`, `subprojects/pyprima/pyprima`, `tools/building/tempita`); all are independently
  permissive (BSD/MIT-style) but are excluded via `exclude_globs` because this pipeline
  attributes one licence per source and cannot record differing per-subtree provenance.
- **scikit-learn**: `sklearn/externals/*` vendors five third-party packages (array_api_compat,
  array_api_extra, _numpydoc, _packaging, _scipy; 57 `.py` files) and is excluded as vendored
  third-party code.
- **Pydantic**: `pydantic-core/*` carries its own MIT `LICENSE` under a different copyright
  holder (Samuel Colvin vs. Pydantic Services Inc.) and is excluded for provenance accuracy
  even though the licence family is compatible.
- **NLTK**: `nltk/corpus/reader/sinica_treebank.py` has an Apache-2.0 header but its docstring
  references a CC BY-NC-SA-licensed corpus; excluded out of caution since the policy's
  `deny_families` includes non-commercial. No other file in the tree matched the same scan.
- **pandas, pytest, FastAPI, SQLAlchemy, Scrapy, aiohttp**: no nested licences with `.py`
  content found; no exclusions needed. Regex hits for "MPL" in pandas/scikit-learn/NetworkX
  were manually confirmed as `matplotlib` import-alias false positives, not the Mozilla Public
  License.
- `path_filters.excluded_parts` in `configs/data/phase2.yaml` also gained `vendored`,
  `externals`, `_vendor`, `third_party`, and `thirdparty` as general-purpose vendoring
  conventions observed across these repositories, for defense-in-depth on future sources.

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

## Production Expansion Results (2026-08-04)

The 11 "safer candidate" libraries identified by the 2026-08-04 metadata-only audit (all
except the large frameworks below) were downloaded, checksum-verified, licence-reviewed for
nested/vendored subtrees, and ingested through the full Phase 2 pipeline. Combined with Click,
the approved corpus totals 4,560 deduplicated pretraining records (34,685,551 UTF-8 bytes;
`data/reports/dataset_report.json`), split 4,304/256/0 across train/validation/test. Splitting
is repository-group-based (98/1/1 ratio); with only 12 groups, none happened to hash into the
1% test bucket for this seed — a structural property of group-isolated splitting with few large
groups, not a bug. Instruction data (Exercism) supplies a non-empty pretraining-independent
test split (see the dataset card).

This 34.7 MB is far short of both the tokenizer's candidate threshold (100 MiB, `configs/
tokenizer/genpy_bpe_16k.yaml: minimum_candidate_bytes`) and its production threshold (500 MiB,
`minimum_production_bytes`) — see `docs/tokenizer_training.md` readiness output. Reaching
either threshold requires either the larger frameworks below or additional smaller libraries;
it is not achievable from the currently prioritized "safer candidates" alone.

### Larger frameworks: still excluded, not part of this expansion

The audit also resolved immutable revisions and inspected complete Git trees for larger
frameworks. `Eligible bytes` is the sum of 64-byte to 1-MiB Python blobs after path-part
exclusions — a planning upper bound before licence-conflict checks, normalization, safety
scanning, quality filtering, and deduplication. None of these were downloaded or approved; they
remain excluded pending an explicit decision to expand scope beyond the prioritized safer
candidates.

| Candidate | Immutable revision | Root licence | Eligible files | Eligible bytes |
| --- | --- | --- | ---: | ---: |
| home-assistant/core | `9f74948e26a653888489eacae14e3fa338fa318e` | Apache-2.0 | 16,981 | 115,038,027 |
| apache/airflow | `b316afb44dc9686f5c39aab678a9a89cb21d53d2` | Apache-2.0 | 7,678 | 62,923,330 |
| pytorch/pytorch | `d47daf6e85dcd5a7ca79ac7b2ca73b1cb113cbd1` | BSD-3-Clause | 4,526 | 101,824,978 |
| tensorflow/tensorflow | `e30ad0b36aaae6d9b7b5a98e811b2e1356f6a117` | Apache-2.0 | 3,035 | 46,050,452 |
| ray-project/ray | `f8aa4d14edf4d58a10e94e86bd31920d1346ca6e` | Apache-2.0 | 4,382 | 46,797,683 |
| saltstack/salt | `c3e5bb1981a025a69bf94269282838ab618079c8` | Apache-2.0 | 2,680 | 33,307,247 |
| huggingface/transformers | `d09f53a801f45ad73ec3510e17972024234bc0fd` | Apache-2.0 | 4,165 | 74,502,970 |
| apache/superset | `e4ef84ca724b6a4de2e916fbbca04a989cbf5d21` | Apache-2.0 | 2,483 | 22,238,823 |

These eight frameworks alone total 502,683,510 eligible raw Python bytes — roughly 16x the
34.7 MB actually retained from the 11 approved safer candidates — which is why they dominate
any realistic path to the 500 MiB production tokenizer-corpus threshold. Approving them was
explicitly out of scope for this expansion (task instructions prioritized the safer candidates
only) and would need the same per-source checksum, nested-licence, and exclusion review applied
above before ingestion.

NumPy, SymPy, and Celery remain excluded because GitHub reports `NOASSERTION` on their licence
detection (unlike NetworkX, these were not manually re-verified in this pass); Django, Apache
Beam, and Dagster require nested-licence resolution; CPython is outside the allowlist;
TheAlgorithms/Python is reserved from training as algorithm/evaluation-like material; and The
Stack v2 remains gated and mixed-licence.
