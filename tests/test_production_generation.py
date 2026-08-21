from pathlib import Path
from collections import Counter

from genpy.data.config import load_data_config
from genpy.data.execution import execute_function_tests
from genpy.data.generation import generate_examples
from genpy.data.validate import apply_validation


def test_production_config_and_generator_are_deterministic() -> None:
    config = load_data_config(Path(__file__).parents[1] / "configs" / "data.yaml")
    assert config.target_examples == 100000
    first = generate_examples(100, seed=42, max_per_family=1)
    second = generate_examples(100, seed=42, max_per_family=1)
    assert [task.example.id for task in first] == [task.example.id for task in second]
    assert len({task.example.family_id for task in first}) == 100
    assert max(Counter(task.example.family_id for task in first).values()) == 1


def test_generated_smoke_tasks_parse_and_pass_semantic_tests() -> None:
    tasks = generate_examples(40, seed=42, max_per_family=1)
    for task in tasks:
        example = task.example
        result = apply_validation(example, strict_code=True)
        assert result.valid
        execution = execute_function_tests(example.response, task.function_name, task.test_cases)
        assert execution.passed, execution.error
