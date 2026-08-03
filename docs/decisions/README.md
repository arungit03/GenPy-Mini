# Architecture Decision Records

This directory holds Architecture Decision Records (ADRs) for GenPy-Mini —
short documents that capture a significant decision, the context behind it,
and its consequences, so future contributors understand *why*, not just
*what*.

## Index

- [ADR-001: Project scope](ADR-001-project-scope.md) — why the first model
  is Python-only, ~33–35M parameters, decoder-only, trained from scratch,
  developed locally but trained on Kaggle, and scoped as an educational /
  portfolio project.
- [ADR-002: Dataset acquisition and licensing](ADR-002-dataset-acquisition-and-licensing.md) —
  why acquisition is separate from cleaning, why unknown licenses are
  blocked or require review, why sources require immutable revisions, and
  why acquisition is resource-bounded and dependency-light.

## Adding a new ADR

Copy the format of ADR-001: number sequentially, state the decision, the
context that drove it, and the consequences (including trade-offs accepted).
ADRs are not updated after the fact to reflect reversals — write a new ADR
that supersedes the old one instead.
