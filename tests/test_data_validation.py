from genpy.data.normalize import normalize_example
from genpy.data.schema import example_from_mapping
from genpy.data.validate import apply_validation, is_obvious_non_python


def make(response: str):
    return normalize_example(example_from_mapping({"id": response[:8], "instruction": "Write Python code", "response": response}))


def test_valid_python_parses() -> None:
    assert apply_validation(make("x = 1\nprint(x)")).valid


def test_syntax_error_is_detected() -> None:
    result = apply_validation(make("def broken(:"), strict_code=True)
    assert not result.valid and not result.syntax_valid


def test_obvious_foreign_languages_are_flagged() -> None:
    assert is_obvious_non_python("#include <stdio.h>")
    assert is_obvious_non_python("public static void main(String[] args) {}")


def test_empty_response_is_rejected() -> None:
    result = apply_validation(make(""), strict_code=True)
    assert not result.valid
