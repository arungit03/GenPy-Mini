"""Transparent static quality scoring for Python source."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from genpy.data.python_validation import PythonValidation


@dataclass(frozen=True, slots=True)
class QualityResult:
    """A score, decision, and inspectable quality components."""

    score: float
    accepted: bool
    reason: str | None
    components: dict[str, float]


def has_generated_marker(text: str, markers: list[str]) -> bool:
    """Detect generated-code notices near the beginning of a file."""
    prefix = "\n".join(text.splitlines()[:20]).lower()
    return any(marker.lower() in prefix for marker in markers)


def _placeholder_ratio(text: str) -> float:
    try:
        tree = ast.parse(text, feature_version=(3, 11))
    except SyntaxError:
        return 1.0
    bodies = [
        node.body
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not bodies:
        return 0.0
    empty = sum(
        len(body) == 1
        and (
            isinstance(body[0], ast.Pass)
            or (
                isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and body[0].value.value is Ellipsis
            )
        )
        for body in bodies
    )
    return empty / len(bodies)


def score_quality(text: str, validation: PythonValidation, config: dict[str, Any]) -> QualityResult:
    """Score syntax, lexical richness, readability, repetition, and completeness."""
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    comments = sum(line.lstrip().startswith("#") for line in nonempty)
    comment_ratio = comments / max(1, len(nonempty))
    long_ratio = sum(len(line) > int(config["maximum_line_length"]) for line in lines) / max(
        1, len(lines)
    )
    repeated_ratio = 0.0
    if nonempty:
        repeated_ratio = 1.0 - len(set(nonempty)) / len(nonempty)
    placeholder_ratio = _placeholder_ratio(text)
    components = {
        "syntax": 1.0 if validation.ast_valid and validation.tokenize_valid else 0.0,
        "lexical": min(
            1.0, validation.lexical_tokens / max(1, int(config["minimum_lexical_tokens"]))
        ),
        "readability": max(
            0.0, 1.0 - long_ratio / max(0.001, float(config["maximum_long_line_ratio"]))
        ),
        "non_repetition": max(
            0.0,
            1.0 - repeated_ratio / max(0.001, float(config["maximum_repeated_line_ratio"])),
        ),
        "completeness": max(0.0, 1.0 - placeholder_ratio),
        "documentation": min(1.0, comment_ratio / 0.15) if comments else 0.5,
    }
    weights = {
        "syntax": 0.35,
        "lexical": 0.15,
        "readability": 0.15,
        "non_repetition": 0.15,
        "completeness": 0.15,
        "documentation": 0.05,
    }
    score = round(sum(components[name] * weight for name, weight in weights.items()), 4)
    reason: str | None = None
    if not validation.ast_valid or not validation.tokenize_valid:
        reason = "invalid_python"
    elif long_ratio > float(config["maximum_long_line_ratio"]):
        reason = "minified_content"
    elif repeated_ratio > float(config["maximum_repeated_line_ratio"]):
        reason = "low_quality"
    elif score < float(config["minimum_score"]):
        reason = "low_quality"
    return QualityResult(score, reason is None, reason, components)
