from genpy.data.deduplicate import deduplicate_examples
from genpy.data.normalize import normalize_example
from genpy.data.schema import example_from_mapping


def record(identifier: str, instruction: str, response: str):
    return normalize_example(example_from_mapping({"id": identifier, "instruction": instruction, "response": response}))


def test_exact_and_instruction_duplicates_are_reported() -> None:
    examples, report = deduplicate_examples([
        record("a", "Say hello", "print('hello')"),
        record("b", "Say hello", "print('hello')"),
    ], near_duplicate=False)
    assert len(examples) == 1
    assert report.exact_duplicates == 1
    assert report.examples_removed == 1


def test_near_duplicate_is_detected_but_distinct_solution_survives() -> None:
    examples, report = deduplicate_examples([
        record("a", "Make a list", "values = [1, 2, 3, 4, 5]\nfor value in values:\n    print(value)\nprint(sum(values))"),
        record("b", "Make a list", "values = [1, 2, 3, 4, 5]\nfor value in values:\n    print(value)\nprint(sum(values) + 0)"),
        record("c", "Make a list", "values = [5, 4, 3, 2, 1]\nfor value in values:\n    print(value * 2)\nprint(sum(values))"),
    ], near_duplicate=True, near_duplicate_threshold=0.90)
    assert len(examples) == 2
    assert report.near_duplicates >= 1 or report.instruction_duplicates >= 1
