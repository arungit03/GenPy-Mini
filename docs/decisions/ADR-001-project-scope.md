# ADR-001: Project Scope

## Status

Accepted — 2026-08-02

## Context

GenPy-Mini is a from-scratch decoder-only Transformer for Python-focused
code generation and related tasks. The available resources are a single
Windows laptop (i7-13620H, 16 GB RAM, integrated GPU only, ~300 GB free
disk) for development, and free-tier Kaggle GPU notebooks for training. A
scope had to be chosen that is achievable within those constraints while
still producing a real, testable system.

## Decision

1. **Python-only language scope.** The model targets Python source code and
   English programming instructions, not a multi-language corpus. This
   keeps the tokenizer vocabulary, dataset size, and evaluation surface
   tractable at small model scale.
2. **~33–35 million parameters.** This is small enough to pretrain on
   free-tier Kaggle GPU sessions in a reasonable number of runs, and small
   enough to run inference on the local machine without a discrete GPU.
   It is not intended to be competitive with commercial coding assistants
   trained on orders of magnitude more compute and data.
3. **Decoder-only architecture.** A single-stack causal Transformer is the
   simplest architecture that supports all six target capabilities
   (generation, completion, debugging, explanation, optimisation, test
   generation) via instruction formatting, without needing a separate
   encoder-decoder pipeline.
4. **Trained from scratch.** No pretrained checkpoint is fine-tuned. This
   keeps licensing simple (the model's weights derive only from data whose
   license is checked, per [docs/data-governance.md](../data-governance.md))
   and makes the project a genuine from-scratch learning exercise.
5. **Local development, Kaggle training.** Coding, testing, and CPU smoke
   tests happen on the local machine, which has no CUDA-capable GPU. Actual
   pretraining and fine-tuning happen on Kaggle, where free GPU time is
   available. Checkpoints must eventually be portable between Kaggle
   sessions (interrupted-session-safety is deferred to Phase 7).
6. **Educational / portfolio scope.** GenPy-Mini is explicitly not expected
   to match commercial coding assistants (e.g. Copilot-class models). Its
   purpose is to demonstrate an end-to-end, from-scratch LLM system —
   data governance, tokenizer, architecture, training, evaluation,
   inference, and a safe serving path — at a scale one engineer can
   actually build, train, and validate.

## Consequences

- Capability and quality will be limited by both model scale and training
  data volume; README and future model cards must not overstate results.
- Multi-language support is out of scope unless a future ADR revisits this
  decision.
- Because training happens on ephemeral Kaggle sessions, checkpoint
  portability and resumability (Phase 7) are a hard requirement, not a nice
  -to-have.
- All generated code must be treated as untrusted output — see the security
  warning in [README.md](../../README.md) and the sandboxing requirement in
  Phase 15 of [docs/roadmap.md](../roadmap.md).
