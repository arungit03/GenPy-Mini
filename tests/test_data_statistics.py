from genpy.data.normalize import normalize_example
from genpy.data.schema import example_from_mapping
from genpy.data.statistics import classify_dataset_size, compute_statistics
from genpy.data.validate import apply_validation


def test_statistics_counts_and_syntax_rate() -> None:
    examples = [normalize_example(example_from_mapping({
        "id": "a", "category": "lists", "task_type": "code_generation",
        "instruction": "List values", "response": "values = [1]\nprint(values)", "source": "curated",
    })), normalize_example(example_from_mapping({
        "id": "b", "category": "functions", "task_type": "code", "code": "def f():\n    return 1", "source": "fixture",
    }))]
    for example in examples:
        apply_validation(example)
    stats = compute_statistics(examples)
    assert stats.total_examples == 2
    assert stats.category_counts == {"lists": 1, "functions": 1}
    assert stats.task_type_counts == {"code_generation": 1, "code": 1}
    assert stats.syntax_valid_percentage == 100.0
    assert classify_dataset_size(10) == "prototype_only"
