# Dataset Policy

## Scope

GenPy V1 data supports beginner-to-intermediate Python generation. Pretraining data and
prompt-to-code instruction data are separate families. Public evaluation benchmarks,
private evaluation prompts, and test fixtures are never training data.

## Scale Plan

| Tier | Purpose | Approximate target |
| --- | --- | ---: |
| Smoke dataset | Local pipeline verification | 10,000-50,000 records |
| GenPy-5M corpus | First model experiment | 50-100M estimated tokens |
| GenPy-25M corpus | Scaling validation | 300-500M estimated tokens |
| GenPy-100M corpus | Final pretraining target | 1-2B actual GenPy tokens |

Phase 2 reports bytes, characters, lines, Python lexical tokens, and a rough subword estimate.
Only Phase 3 can produce exact GenPy token counts.

## Licence And Provenance

The provisional allowlist is MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, 0BSD,
CC0-1.0, and Unlicense. Unknown, custom/unreviewed, conflicting, copyleft, MPL,
non-commercial, and research-only licences are excluded. Dataset-level metadata cannot
replace repository/file-level metadata for mixed-source code. This is project policy, not
legal advice or a guarantee of legal safety.

Each accepted record retains source ID and URL, repository, immutable revision, original
path, SPDX identifier, and content hash. Validated removals go in
`configs/data/opt_out.yaml` before rebuilding. Upstream opt-out requirements also apply.

## Filtering

The pipeline preserves indentation, comments, and docstrings. It normalizes UTF-8 and line
endings, removes trailing whitespace, and enforces one final newline. It excludes binary,
null-containing, malformed, generated, minified, vendored, incomplete, and out-of-size
files. Tests are eligible when they pass the same configurable rules.

Secret, PII, and unsafe-code scanners use static conservative patterns. Quarantine manifests
contain categories and hashes, never matched text. Scanner controls currently use five
fictional safe records; their observed false-positive result is emitted in each build report.
No scanner guarantees complete removal, and findings require periodic human review.

## Quality Score

The score is a weighted sum from 0 to 1 and does not reward file size:

| Component | Weight | Meaning |
| --- | ---: | --- |
| Syntax | 0.35 | `ast.parse` and `tokenize` both succeed |
| Lexical richness | 0.15 | Configured minimum lexical-token threshold |
| Readability | 0.15 | Long-line ratio below threshold |
| Non-repetition | 0.15 | Repeated non-empty line ratio below threshold |
| Completeness | 0.15 | Function/class bodies are not mostly placeholders |
| Documentation | 0.05 | Modest comment signal without requiring comments |

Thresholds live in `configs/data/phase2.yaml`. Sensitive imports are flagged for review but
are not treated as proof of unsafe intent.

## Splitting And Leakage

Exact and near deduplication occurs globally before splitting. Pretraining uses 98/1/1 and
instruction data uses 80/10/10. A seeded hash assigns whole repositories/projects or problem
families to one split. Exact cross-split duplicates fail validation; near-duplicate and group
checks are included in the leakage report. The final private execution benchmark remains
outside this dataset.
