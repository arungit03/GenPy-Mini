"""Structural, syntax, and lightweight quality validation."""

import ast
import re
from dataclasses import dataclass, field

from .schema import CodeExample, InstructionExample

FOREIGN_MARKERS = (
    "#include <", "public static void main", "system.out.println", "std::cout",
    "console.log(", "package main", "func main()",
)
PLACEHOLDER_MARKERS = ("solution omitted", "write code here", "placeholder answer")


@dataclass
class ValidationResult:
    valid: bool
    syntax_valid: bool = False
    non_python_suspected: bool = False
    issues: list[str] = field(default_factory=list)


def code_for(example: InstructionExample | CodeExample) -> str:
    return example.code if isinstance(example, CodeExample) else example.response


def is_obvious_non_python(code: str) -> bool:
    lowered = code.lower()
    return any(marker in lowered for marker in FOREIGN_MARKERS) or "<html" in lowered


def validate_example(example: InstructionExample | CodeExample, strict_code: bool = False) -> ValidationResult:
    issues: list[str] = []
    try:
        example.validate()
    except (TypeError, ValueError) as exc:
        issues.append(str(exc))
    instruction = getattr(example, "instruction", "")
    code = code_for(example)
    if hasattr(example, "instruction") and not instruction.strip():
        issues.append("empty instruction")
    if not isinstance(code, str) or len(code.strip()) < 3:
        issues.append("response/code is too short")
        code = str(code)
    non_python = is_obvious_non_python(code)
    if non_python:
        issues.append("obvious non-Python syntax detected")
    lowered = code.lower()
    if any(marker in lowered for marker in PLACEHOLDER_MARKERS) or re.search(r"\btodo\b", lowered):
        issues.append("placeholder or TODO response")
    syntax_valid = False
    try:
        ast.parse(code)
        syntax_valid = True
    except (SyntaxError, ValueError, TypeError):
        issues.append("Python syntax error")
    if strict_code and not syntax_valid:
        pass
    valid = not issues and (not strict_code or syntax_valid)
    return ValidationResult(valid, syntax_valid, non_python, issues)


def apply_validation(example: InstructionExample | CodeExample, strict_code: bool = False) -> ValidationResult:
    result = validate_example(example, strict_code=strict_code)
    example.syntax_valid = result.syntax_valid
    return result
