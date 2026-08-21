from genpy.evaluation.coding import evaluate_code, extract_python, generation_summary, run_function_tests_subprocess


def test_code_extraction_and_syntax_are_separate() -> None:
    assert extract_python("Here is the answer:\n```python\ndef solve(x):\n    return x + 1\n```") == "def solve(x):\n    return x + 1"
    assert evaluate_code("not a Python answer")['python_extracted'] is False
    assert evaluate_code("def solve(x):\n    return x + 1", {"test_cases": [{"args": [2], "expected": 3}]})["functional_correct"] is True


def test_functional_timeout_and_duplicate_metrics() -> None:
    timeout = run_function_tests_subprocess("def solve(x):\n    while True: pass", "solve", [{"args": [1], "expected": 1}], timeout_seconds=0.1)
    assert timeout["category"] == "timeout"
    summary = generation_summary(["same", "same", "different"])
    assert summary["exact_duplicate_generations"] == 1 and summary["unique_output_rate"] == 2 / 3
