"""Deterministic, semantically tested Python task generation."""

from .registry import generate_examples, generate_smoke_examples

__all__ = ["generate_examples", "generate_smoke_examples"]
