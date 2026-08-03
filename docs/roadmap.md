# GenPy-Mini Roadmap

Each phase lists its deliverable and the gate that must pass before the next
phase begins. Phases are sequential; later phases may reveal that an earlier
gate needs revisiting, which is expected.

## Phase 1 — Foundation (complete)

- **Deliverable**: repository structure, documented target architecture,
  validated configuration system, development standards, CI quality
  workflow, and environment validation script.
- **Acceptance gate**: `pytest`, `ruff`, and `mypy` pass; the model
  configuration loads and validates; no dataset, tokenizer, model, or
  training code exists; no heavy ML dependency is present.
- **Status**: complete.

## Phase 2 — Dataset acquisition and governance (complete)

- **Deliverable**: a documented, tooling-enforced process for registering,
  reviewing, and reproducibly acquiring licensed Python source material
  (`local_directory`, `git_repository`, `http_archive`), with full
  provenance manifests and audit reports. See
  [dataset-acquisition.md](dataset-acquisition.md).
- **Acceptance gate**: every acquired source has a recorded license and
  governance decision; unknown or unapproved sources are refused by
  default; data-governance rules in
  [docs/data-governance.md](data-governance.md) are enforced by tooling
  (`src/genpy/data/`), not just documentation; all Phase 2 tests, Ruff,
  and mypy pass; acquired data is explicitly labeled raw and
  not-yet-training-ready.
- **Status**: complete. Cleaning, secret/PII scanning, deduplication,
  quality filtering, and repository-level splitting are explicitly **not**
  implemented here — see Phase 3.

## Phase 3 — Dataset cleaning and deduplication (next)

- **Deliverable**: cleaning pipeline that strips secrets, personal data,
  generated/vendored code, and duplicate content; splits data by repository.
- **Acceptance gate**: a dataset report is produced showing before/after
  counts and no train/test leakage across repository boundaries.

## Phase 4 — Tokenizer

- **Deliverable**: a Python-focused tokenizer trained on the cleaned corpus,
  targeting the 16,000-token vocabulary in `config/model_config.yaml`.
- **Acceptance gate**: round-trip encode/decode tests pass; special tokens
  match the configuration; vocabulary size matches the target.

## Phase 5 — Tokenized data pipeline

- **Deliverable**: batched, shuffled, checkpoint-resumable sequence loader
  producing fixed-length (512-token) training examples.
- **Acceptance gate**: throughput and memory are validated on CPU at small
  scale; pipeline output shape matches model input expectations.

## Phase 6 — GenPy-Nano architecture test

- **Deliverable**: a tiny (sub-1M parameter) decoder-only Transformer used
  purely to validate the architecture code end-to-end on CPU.
- **Acceptance gate**: forward and backward passes run on CPU without error
  on a small synthetic batch; parameter count matches expectations.

## Phase 7 — Kaggle training pipeline

- **Deliverable**: a Kaggle-notebook-compatible training script with
  interrupted-session-safe checkpointing.
- **Acceptance gate**: a short training run on Kaggle completes, checkpoints,
  and resumes correctly after a simulated interruption.

## Phase 8 — GenPy-Tiny validation

- **Deliverable**: a mid-scale model (a few million parameters) trained
  briefly on Kaggle to validate the full pipeline before committing GPU time
  to the full-size run.
- **Acceptance gate**: loss decreases as expected; checkpoint is portable
  back to another Kaggle session.

## Phase 9 — GenPy-Mini pretraining

- **Deliverable**: base pretraining of the full 33–35M parameter GenPy-Mini
  model on the tokenized Python corpus.
- **Acceptance gate**: training completes to a defined step/loss target;
  final parameter count falls within the configured target range.

## Phase 10 — Instruction dataset

- **Deliverable**: an instruction-formatted dataset covering the target
  capabilities (generation, completion, debugging, explanation,
  optimisation, test generation), using the special tokens already defined
  in `config/model_config.yaml`.
- **Acceptance gate**: dataset passes the same governance checks as Phase 2/3
  and covers all six target capabilities.

## Phase 11 — Instruction tuning

- **Deliverable**: fine-tuning of the pretrained base model on the
  instruction dataset.
- **Acceptance gate**: qualitative and automated checks show instruction
  -following behaviour improves over the base model on held-out examples.

## Phase 12 — Evaluation

- **Deliverable**: an evaluation harness covering the six target
  capabilities with reproducible metrics.
- **Acceptance gate**: evaluation runs end-to-end and produces a versioned
  report; no unsubstantiated performance claims are made.

## Phase 13 — Inference

- **Deliverable**: a local/offline inference path (batching, sampling
  strategy, stopping criteria) built on the tuned checkpoint.
- **Acceptance gate**: inference produces syntactically valid Python for a
  fixed smoke-test prompt set.

## Phase 14 — FastAPI service

- **Deliverable**: an HTTP API exposing the target capabilities.
- **Acceptance gate**: API starts, passes basic integration tests, and
  documents its request/response schema.

## Phase 15 — Secure sandbox

- **Deliverable**: an isolated execution environment for running
  model-generated code safely (resource limits, no network, no host
  filesystem access).
- **Acceptance gate**: sandbox blocks a documented set of known-dangerous
  operations in automated tests before any generated code executes there.

## Phase 16 — Frontend

- **Deliverable**: a web UI for interacting with the target capabilities.
- **Acceptance gate**: UI covers all six target capabilities against the
  Phase 14 API in a manual end-to-end pass.

## Phase 17 — Deployment and model card

- **Deliverable**: deployment configuration and a model card documenting
  training data provenance, intended use, limitations, and known risks.
- **Acceptance gate**: model card is published alongside the deployed
  service; deployment is reproducible from a clean environment.
