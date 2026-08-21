import pytest

from genpy.data.schema import InstructionExample, example_from_mapping


def test_valid_instruction_example_is_accepted() -> None:
    example = InstructionExample("x", "code_generation", "beginner", "Say hi", "print('hi')")
    example.validate()


def test_missing_required_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="instruction"):
        example_from_mapping({"id": "x", "response": "print(1)"}).validate()


def test_invalid_quality_is_rejected() -> None:
    with pytest.raises(ValueError, match="quality_score"):
        example_from_mapping({"id": "x", "instruction": "Say hi", "response": "print(1)", "quality_score": 2}).validate()


def test_invalid_task_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="task_type"):
        example_from_mapping({"id": "x", "task_type": "chat", "category": "beginner", "instruction": "Say hi", "response": "print(1)"}).validate()
