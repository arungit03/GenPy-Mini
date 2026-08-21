# GenPy Python 100K Dataset Card

## Purpose and status

`GenPy-Python-100K` is a 100,000-row Python instruction-to-code corpus for later GenPy pretraining and instruction work. Checkpoint 2.5 is complete. The corpus is programmatically generated from versioned task templates; it is not human-written or imported repository code.

## Provenance and licensing

The production rows use `source=genpy_programmatic` and `license=generated`. No unknown-license data is included. Optional CodeSearchNet support exists only for reviewed local records with explicit license metadata and was not used in this build. Benchmark sources are kept separate and no benchmark rows were ingested.

## Methodology

The builder creates meaningful parameterized task instances spanning beginner Python, conditions, functions, strings, collections, OOP, data structures, algorithms, debugging, completion, files, NumPy, Pandas, and intermediate Python. Each task has a deterministic ID, family ID, template ID, variant metadata, difficulty, and interface metadata. The effective family cap is one parameterized instance per family, a conservative choice that prevents paraphrase leakage and near-clone accumulation.

The generated response is normalized, parsed with `ast.parse`, and executed against deterministic semantic test cases in a controlled namespace. Invalid syntax, execution failures, secrets, benchmark matches, and length violations are rejected before deduplication and splitting. Exact and near-duplicate checks remain enabled; explicit family boundaries are respected for parameterized generated tasks because they already define the leakage unit.

## Schema

Rows use the Checkpoint 2 `InstructionExample` schema: `id`, `task_type`, `category`, `instruction`, `input`, `response`, `language`, `source`, `quality_score`, `metadata`, `family_id`, and `syntax_valid`. Metadata records difficulty, template/variant IDs, interface, execution status, test count, and generated license provenance.

## Composition

The final corpus has 100,000 instruction rows, 100,000 unique families, and a maximum of one row per family. Category and task-type counts are recorded in `reports/python_100k_categories.json`. Difficulty is heuristic and targets approximately 45% easy, 40% medium, and 15% hard.

## Splits and leakage

The split is deterministic with seed 42 and uses whole-family assignment: 90,000 train, 5,000 validation, and 5,000 test. Family overlap is zero. Repository overlap is zero because no external repositories were imported.

## Validation and reproducibility

All final rows are syntax-valid and execution-tested with a 100% execution pass rate. SHA-256 hashes for the clean corpus, splits, configuration, and source manifest are in `reports/python_100k_hashes.json`; the full build manifest is `data/production/MANIFEST.json`. Rebuilding with the same generator, configuration, and seed produces stable rows and split assignments.

## Intended use

The dataset is intended for Python-focused language-model research after the tokenizer and model checkpoints are implemented. It is not a benchmark, not a claim of human-quality equivalence, and not evidence that GenPy has been trained.

## Limitations

Programmatic examples provide broad, controlled coverage but do not replace human review or verified external code diversity. Difficulty is heuristic, third-party library coverage is intentionally small, and semantic family grouping is based on the generator's explicit parameter families. No tokenizer statistics are included because tokenization is Checkpoint 3.
