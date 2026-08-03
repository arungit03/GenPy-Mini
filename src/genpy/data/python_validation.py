"""Safe Python syntax and lexical validation without importing code."""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PythonValidation:
    """Static Python validation results."""

    ast_valid: bool
    tokenize_valid: bool
    lexical_tokens: int
    error: str | None
    review_flags: tuple[str, ...]


def validate_python(text: str) -> PythonValidation:
    """Parse and tokenize Python 3.11-compatible source without execution."""
    try:
        tree = ast.parse(text, feature_version=(3, 11))
    except SyntaxError as error:
        return PythonValidation(False, False, 0, f"SyntaxError:{error.lineno}", ())
    count = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type not in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.NEWLINE,
                tokenize.NL,
            }:
                count += 1
    except (IndentationError, tokenize.TokenError) as error:
        return PythonValidation(True, False, count, type(error).__name__, ())
    review_modules = {
        "ctypes",
        "marshal",
        "os",
        "pickle",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "urllib",
    }
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.partition(".")[0]
                if root in review_modules:
                    flags.add(f"import:{root}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.partition(".")[0]
            if root in review_modules:
                flags.add(f"import:{root}")
    return PythonValidation(True, True, count, None, tuple(sorted(flags)))
