from genpy.data.normalize import normalize_example
from genpy.data.schema import example_from_mapping


def test_normalization_preserves_indentation_and_cleans_text() -> None:
    example = example_from_mapping({"id": "x", "category": "linked-list", "instruction": "  Make a node  ", "response": "\r\n```python\r\nif x:\r\n    print(x)\r\n```\r\n"})
    normalized = normalize_example(example)
    assert normalized.category == "linked_lists"
    assert normalized.response == "if x:\n    print(x)"


def test_null_bytes_are_removed() -> None:
    example = example_from_mapping({"id": "x", "instruction": "Say hi", "response": "print(\x00'hi')"})
    assert "\x00" not in normalize_example(example).response
