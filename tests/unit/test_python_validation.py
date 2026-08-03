from __future__ import annotations

from genpy.data.python_validation import validate_python


def test_valid_and_invalid_python() -> None:
    valid = validate_python("def square(value: int) -> int:\n    return value * value\n")
    invalid = validate_python("def square(:\n    pass\n")
    assert valid.ast_valid and valid.tokenize_valid and valid.lexical_tokens > 0
    assert not invalid.ast_valid


def test_sensitive_import_is_flagged_for_review_without_execution() -> None:
    result = validate_python("import subprocess\nprint('review only')\n")
    assert result.ast_valid
    assert result.review_flags == ("import:subprocess",)
