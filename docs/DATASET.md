# GenPy Python Dataset Pipeline

## Purpose and status

Checkpoint 2 prepares Python instruction-to-code and Python code examples for later training. The pipeline is complete and tested, but no production dataset is claimed or bundled. The target is approximately 100,000 high-quality examples; a small fixture is used for smoke verification.

## Schema

Instruction records use `id`, `task_type`, `category`, `instruction`, `input`, `response`, `language`, `source`, `quality_score`, `metadata`, `family_id`, and `syntax_valid`. Code-only records use `code` instead of `instruction`/`response`. Required semantic fields are validated, and quality scores must be in `[0.0, 1.0]`.

Supported task types include code generation, completion, bug fixing, explanation, optimization, algorithms, data structures, library usage, and `code`. Categories are centrally registered; aliases such as `linked-list` normalize to `linked_lists`.

## Preparation

```text
JSON/JSONL input → schema parse → conservative normalization → validation/filtering
→ deterministic IDs/family IDs → exact/near deduplication → atomic JSONL output
```

Normalization fixes line endings, null bytes, outer whitespace, and single Python code fences. It does not rewrite Python indentation. `ast.parse` validates code when strict mode is enabled. Lightweight checks flag obvious Java, C/C++, Go, HTML, placeholder, and TODO content.

Run the pipeline with:

```bash
python scripts/prepare_python_data.py --input data/raw/python_examples.jsonl --output data/processed/python/all_clean.jsonl --strict-code --deduplicate --report
```

## Deduplication and leakage prevention

Exact hashes cover instruction/input/response content. Identical instructions and code are reported separately. A conservative token-shingle Jaccard check is available with a default threshold of `0.90`. The first deterministic occurrence is retained.

Examples are assigned explicit `family_id` values when supplied. Otherwise a deterministic heuristic recognizes a few obvious families and falls back to an instruction fingerprint. This is useful leakage protection, not semantic clustering. Splits assign whole families to exactly one of train, validation, or test.

## Splits and statistics

`build_python_splits.py` defaults to 90% train, 5% validation, and 5% test with seed 42. Whole-family assignment can make tiny datasets differ slightly from exact ratios. Statistics use characters and lines only; token counts belong to Checkpoint 3.

The validated defaults live in `configs/data.yaml` and are loaded with `genpy.data.config.load_data_config`.

## Quality, provenance, and benchmarks

Automatic quality scoring is intentionally simple: supplied scores are preserved and validated; missing scores default to `1.0`. Sources and license notes belong in `data/manifests/sources.json`. Benchmark exclusion support is provided by `ExclusionRegistry`; the manifest calls out HumanEval, MBPP, APPS, and CodeContests. No benchmark data is downloaded by this checkpoint.

## Output and reproducibility

JSONL output is UTF-8, newline-terminated, sorted by stable keys, written through an atomic same-directory replacement, and hashed with SHA-256. Rebuilding with identical inputs and options preserves record order, IDs, deduplication, and family-grouped split assignments.

## Limitations

The foreign-language detector is heuristic, near-duplicate detection is conservative, and fallback family grouping cannot understand arbitrary semantic equivalence. Production ingestion still requires source/license review, benchmark exclusion data, human quality review, and enough verified examples to approach the 100K target.
