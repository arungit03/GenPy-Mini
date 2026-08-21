"""Registry of meaningful parameterized Python task families."""

import hashlib
from typing import Any

from .base import GeneratedTask, difficulty_for
from ..schema import InstructionExample


CATEGORY_FAMILY_COUNTS = {
    "beginner": 560, "conditions": 320, "functions": 360, "strings": 400,
    "oop": 280, "arrays": 120, "linked_lists": 100, "stacks": 80, "queues": 80,
    "trees": 70, "graphs": 70, "algorithms": 480, "debugging": 320,
    "code_completion": 200, "files": 160, "numpy": 120, "pandas": 120,
    "intermediate_python": 120, "misc": 40,
}

BLUEPRINTS = {
    "beginner": ["parity", "sign", "digit_sum", "reverse"],
    "conditions": ["parity", "sign", "prime"],
    "functions": ["factorial", "fibonacci", "safe_int", "generator"],
    "strings": ["reverse", "vowels", "unique", "anagram"],
    "oop": ["accumulator", "counter", "dataclass_like"],
    "arrays": ["unique", "chunk", "sort"],
    "linked_lists": ["linked_values", "reverse"],
    "stacks": ["stack"],
    "queues": ["queue"],
    "trees": ["tree_sum", "binary_search"],
    "graphs": ["graph_degree", "bfs_order"],
    "algorithms": ["sort", "binary_search", "window", "prefix_sum", "merge"],
    "debugging": ["bug_fix", "bug_boundary", "bug_accumulator"],
    "code_completion": ["completion", "completion_loop"],
    "files": ["file_lines", "safe_int", "csv_rows"],
    "numpy": ["numpy_mean", "numpy_shape", "numpy_filter"],
    "pandas": ["pandas_filter", "pandas_group", "pandas_missing"],
    "intermediate_python": ["generator", "merge", "decorator_like", "context_like"],
    "misc": ["chunk", "safe_int", "merge"],
}


INSTRUCTION_FORMS = (
    "Write a Python function that {description}.",
    "Implement Python code to {description}.",
    "Create a reusable Python solution that {description}.",
    "Given the input values, use Python to {description}.",
)


def _stable_id(family_id: str, variant: int, response: str) -> str:
    digest = hashlib.sha256(f"{family_id}\n{variant}\n{response}".encode()).hexdigest()[:20]
    return f"py_cgen_{digest}"


def _task(category: str, blueprint: str, family_index: int, variant: int,
          description: str, code: str, tests: list[dict[str, Any]],
          task_type: str = "code_generation", difficulty: str | None = None,
          extra: dict[str, Any] | None = None) -> GeneratedTask:
    family_id = f"{category}/{blueprint}/{family_index:04d}"
    metadata = {
        "difficulty": difficulty or difficulty_for(family_index + variant),
        "template_id": f"{category}_{blueprint}_v1",
        "variant_id": f"variant_{variant:02d}",
        "family_id": family_id,
        "interface": "function",
        "execution_tested": False,
        "execution_passed": False,
        "test_count": len(tests),
        "license": "generated",
    }
    if extra:
        metadata.update(extra)
    parameter_note = f" for the {category} parameter set {family_index + 2}"
    instruction = INSTRUCTION_FORMS[variant % len(INSTRUCTION_FORMS)].format(description=description + parameter_note)
    example = InstructionExample(
        id=_stable_id(family_id, variant, code), task_type=task_type, category=category,
        instruction=instruction, response=code, source="genpy_programmatic",
        quality_score=1.0, metadata=metadata, family_id=family_id,
    )
    return GeneratedTask(example=example, function_name="solve", test_cases=tests)


def render_task(category: str, blueprint: str, family_index: int, variant: int) -> GeneratedTask:
    """Render one family member with meaningful parameter and implementation changes."""
    p = 2 + (family_index % 19)
    mode = variant % 3
    if blueprint == "parity":
        divisor = 2 + family_index + variant
        code = f'def solve(value):\n    return "divisible" if value % {divisor} == 0 else "not divisible"'
        tests = [{"args": [divisor], "expected": "divisible"}, {"args": [divisor + 1], "expected": "not divisible"}]
        return _task(category, blueprint, family_index, variant, f"check whether a number is divisible by {divisor}", code, tests)
    if blueprint == "sign":
        code = f'def solve(value, boundary={p}):\n    return "above" if value > boundary else "below" if value < boundary else "equal"'
        return _task(category, blueprint, family_index, variant, f"compare an integer with the boundary {p}", code, [{"args": [p + 1], "expected": "above"}, {"args": [p], "expected": "equal"}])
    if blueprint == "digit_sum":
        code = f'def solve(value):\n    scale = {p}\n    return sum(int(digit) for digit in str(abs(value))) * scale'
        digit_value = p * 101
        digit_expected = sum(int(digit) for digit in str(abs(digit_value))) * p
        return _task(category, blueprint, family_index, variant, f"return the digit sum scaled by {p}", code, [{"args": [digit_value], "expected": digit_expected}])
    if blueprint == "factorial":
        if mode == 0:
            code = f'def solve(value):\n    if value > {p}:\n        raise ValueError("value exceeds configured maximum")\n    result = 1\n    for number in range(2, value + 1):\n        result *= number\n    return result'
        else:
            code = f'def solve(value):\n    if value > {p}:\n        raise ValueError("value exceeds configured maximum")\n    if value < 2:\n        return 1\n    return value * solve(value - 1)'
        return _task(category, blueprint, family_index, variant, f"return factorial values up to the configured maximum {p}", code, [{"args": [p % 8], "expected": __import__("math").factorial(p % 8)}])
    if blueprint == "fibonacci":
        code = f'def solve(count):\n    if count > {p}:\n        raise ValueError("count exceeds configured maximum")\n    values = []\n    first, second = 0, 1\n    for _ in range(count):\n        values.append(first)\n        first, second = second, first + second\n    return values'
        n = min(p, 3 + (p % 5))
        expected = [0, 1, 1, 2, 3, 5, 8][:n]
        return _task(category, blueprint, family_index, variant, "return the first n Fibonacci numbers", code, [{"args": [n], "expected": expected}])
    if blueprint == "prime":
        code = f'def solve(value):\n    maximum = {p}\n    if value < 2 or value > maximum:\n        return False\n    for divisor in range(2, int(value ** 0.5) + 1):\n        if value % divisor == 0:\n            return False\n    return True'
        return _task(category, blueprint, family_index, variant, "determine whether an integer is prime", code, [{"args": [2], "expected": True}, {"args": [p * 2], "expected": False}])
    if blueprint == "reverse":
        code = f'def solve(text):\n    limit = {p}\n    return text[::-1][:limit]' if mode else f'def solve(text):\n    limit = {p}\n    return "".join(reversed(text))[:limit]'
        return _task(category, blueprint, family_index, variant, f"reverse a string and keep at most {p} characters", code, [{"args": ["python"], "expected": "nohtyp"[:p]}])
    if blueprint == "vowels":
        allowed = "aeiou" if p % 2 else "aeiouy"
        code = f'def solve(text):\n    allowed = {allowed!r}\n    maximum = {p}\n    return min(sum(character.lower() in allowed for character in text), maximum)'
        return _task(category, blueprint, family_index, variant, f"count vowels using alphabet {allowed} with cap {p}", code, [{"args": ["GenPy"], "expected": 2 if "y" in allowed else 1}])
    if blueprint == "unique":
        code = f'def solve(values):\n    minimum = {p}\n    return list(dict.fromkeys(value for value in values if value >= minimum))'
        values = [p, p + 1, p, p + 2, p + 1]
        return _task(category, blueprint, family_index, variant, f"remove duplicates while retaining values at least {p}", code, [{"args": [values], "expected": [p, p + 1, p + 2]}])
    if blueprint == "anagram":
        code = f'def solve(left, right):\n    ignored = str({p})\n    left = left.replace(ignored, "")\n    right = right.replace(ignored, "")\n    return sorted(left.replace(" ", "").lower()) == sorted(right.replace(" ", "").lower())'
        return _task(category, blueprint, family_index, variant, f"check whether two strings are anagrams after ignoring the marker {p}", code, [{"args": ["listen", "silent"], "expected": True}])
    if blueprint == "chunk":
        size = 2 + (p % 4)
        code = f'def solve(values):\n    return [values[index:index + {size}] for index in range(0, len(values), {size})]'
        values = list(range(size + 2))
        return _task(category, blueprint, family_index, variant, f"split a list into chunks of size {size}", code, [{"args": [values], "expected": [values[:size], values[size:]]}])
    if blueprint == "sort":
        reverse = mode == 1
        code = f'def solve(values):\n    limit = {p}\n    return sorted(values, reverse={reverse})[:limit]'
        values = [p, 1, p - 1, 3]
        return _task(category, blueprint, family_index, variant, f"return at most {p} values in sorted order", code, [{"args": [values], "expected": sorted(values, reverse=reverse)[:p]}], "algorithm_implementation")
    if blueprint == "binary_search":
        code = f'def solve(values, target):\n    left, right = 0, min(len(values) - 1, {p})\n    while left <= right:\n        middle = (left + right) // 2\n        if values[middle] == target:\n            return middle\n        if values[middle] < target:\n            left = middle + 1\n        else:\n            right = middle - 1\n    return -1'
        return _task(category, blueprint, family_index, variant, "find the index of a target in a sorted list using binary search", code, [{"args": [[1, 3, 5, 7], 5], "expected": 2}], "algorithm_implementation")
    if blueprint == "window":
        width = 2 + (p % 3)
        code = f'def solve(values):\n    limit = max({p}, {width})\n    values = values[:limit]\n    return max(sum(values[index:index + {width}]) for index in range(len(values) - {width} + 1))'
        values = list(range(1, width + 2))
        expected_values = values[:max(p, width)]
        expected = max(sum(expected_values[i:i + width]) for i in range(len(expected_values) - width + 1))
        return _task(category, blueprint, family_index, variant, f"find the maximum sum of a window of width {width}", code, [{"args": [values], "expected": expected}], "algorithm_implementation")
    if blueprint == "prefix_sum":
        code = f'def solve(values):\n    limit = {p}\n    totals = []\n    running = 0\n    for value in values[:limit]:\n        running += value\n        totals.append(running)\n    return totals'
        return _task(category, blueprint, family_index, variant, f"return prefix sums for at most {p} values", code, [{"args": [[1, 2, 3]], "expected": [1, 3, 6][:p]}], "algorithm_implementation")
    if blueprint == "merge":
        code = f'def solve(left, right):\n    key_limit = {p}\n    result = dict(left)\n    result.update(right)\n    return {{key: value for key, value in result.items() if len(str(key)) <= key_limit}}'
        return _task(category, blueprint, family_index, variant, f"merge dictionaries while retaining keys of at most {p} characters", code, [{"args": [{"a": 1}, {"b": 2}], "expected": {"a": 1, "b": 2}}])
    if blueprint == "safe_int":
        code = f'def solve(value, default={p}):\n    try:\n        return int(value)\n    except (TypeError, ValueError):\n        return default'
        return _task(category, blueprint, family_index, variant, f"parse an integer safely with fallback {p}", code, [{"args": ["12"], "expected": 12}, {"args": ["bad"], "expected": p}], "library_usage")
    if blueprint == "stack":
        code = f'def solve(values):\n    capacity = {p}\n    stack = []\n    for value in values[:capacity]:\n        stack.append(value)\n    result = []\n    while stack:\n        result.append(stack.pop())\n    return result'
        return _task(category, blueprint, family_index, variant, f"reverse up to {p} values using a stack", code, [{"args": [[1, 2, 3]], "expected": list(reversed([1, 2, 3][:p]))},], "data_structure")
    if blueprint == "queue":
        code = f'from collections import deque\n\ndef solve(values):\n    capacity = {p}\n    queue = deque(values[:capacity])\n    return [queue.popleft() for _ in range(len(queue))]'
        return _task(category, blueprint, family_index, variant, f"process up to {p} values in first-in first-out order", code, [{"args": [[1, 2, 3]], "expected": [1, 2, 3][:p]}], "data_structure")
    if blueprint in {"accumulator", "counter", "dataclass_like"}:
        code = f'class Accumulator:\n    def __init__(self, start={p}):\n        self.value = start\n\n    def add(self, amount):\n        self.value += amount\n        return self.value\n\ndef solve(values):\n    accumulator = Accumulator()\n    for value in values:\n        accumulator.add(value)\n    return accumulator.value'
        return _task(category, blueprint, family_index, variant, f"use a class to accumulate values starting at {p}", code, [{"args": [[1, 2, 3]], "expected": p + 6}], "data_structure")
    if blueprint == "linked_values":
        code = f'class Node:\n    def __init__(self, value, next_node=None):\n        self.value = value\n        self.next = next_node\n\ndef solve(values):\n    capacity = {p}\n    head = None\n    for value in reversed(values[:capacity]):\n        head = Node(value, head)\n    result = []\n    while head is not None:\n        result.append(head.value)\n        head = head.next\n    return result'
        return _task(category, blueprint, family_index, variant, f"build and traverse a linked list with capacity {p}", code, [{"args": [[1, 2, 3]], "expected": [1, 2, 3][:p]}], "data_structure")
    if blueprint == "tree_sum":
        code = f'def solve(tree, depth={p}):\n    if tree is None or depth < 0:\n        return 0\n    value, left, right = tree\n    return value + solve(left, depth - 1) + solve(right, depth - 1)'
        return _task(category, blueprint, family_index, variant, f"recursively sum a binary tree up to depth {p}", code, [{"args": [[2, [1, None, None], [3, None, None]]], "expected": 6}], "data_structure")
    if blueprint == "graph_degree":
        code = f'def solve(edges):\n    degree_limit = {p}\n    degrees = {{}}\n    for left, right in edges[:degree_limit]:\n        degrees[left] = degrees.get(left, 0) + 1\n        degrees[right] = degrees.get(right, 0) + 1\n    return degrees'
        return _task(category, blueprint, family_index, variant, f"count graph degrees for up to {p} edges", code, [{"args": [[["a", "b"], ["a", "c"]]], "expected": {"a": 2, "b": 1, "c": 1}}], "data_structure")
    if blueprint == "bfs_order":
        code = f'from collections import deque\n\ndef solve(graph, start):\n    node_limit = {p}\n    seen = {{start}}\n    queue = deque([start])\n    order = []\n    while queue and len(order) < node_limit:\n        node = queue.popleft()\n        order.append(node)\n        for neighbor in graph.get(node, []):\n            if neighbor not in seen:\n                seen.add(neighbor)\n                queue.append(neighbor)\n    return order'
        return _task(category, blueprint, family_index, variant, f"traverse at most {p} graph nodes in breadth-first order", code, [{"args": [{"a": ["b"], "b": ["c"], "c": []}, "a"], "expected": ["a", "b", "c"][:p]}], "algorithm_implementation")
    if blueprint == "bug_fix":
        code = f'def solve(value):\n    return value % {p} == 0'
        return _task(category, blueprint, family_index, variant, f"fix the divisibility comparison for {p}", code, [{"args": [p], "expected": True}, {"args": [p + 1], "expected": False}], "bug_fixing", extra={"bug_type": "wrong_comparison_operator", "original_correct_hash": hashlib.sha256(code.encode()).hexdigest()})
    if blueprint == "bug_boundary":
        code = f'def solve(values):\n    result = {p}\n    return result - {p} + sum(values)'
        return _task(category, blueprint, family_index, variant, f"fix an accumulator initialized at {p}", code, [{"args": [[1, 2, 3]], "expected": 6}], "bug_fixing", extra={"bug_type": "incorrect_accumulator"})
    if blueprint == "bug_accumulator":
        code = f'def solve(values):\n    result = {p} - {p - 1}\n    for value in values:\n        result *= value\n    return result'
        return _task(category, blueprint, family_index, variant, f"repair a product accumulator with configuration {p}", code, [{"args": [[2, 3, 4]], "expected": 24}], "bug_fixing", extra={"bug_type": "incorrect_initialization"})
    if blueprint in {"completion", "completion_loop"}:
        code = f'def solve(values):\n    result = []\n    for value in values[:{p}]:\n        result.append(value * value)\n    return result'
        return _task(category, blueprint, family_index, variant, f"complete a function that squares up to {p} values", code, [{"args": [[1, 2, 3]], "expected": [1, 4, 9][:p]}], "code_completion", extra={"completion_prefix": "def solve(values):\n    result = []"})
    if blueprint == "file_lines":
        code = f'def solve(text):\n    limit = {p}\n    return min(len(text.splitlines()), limit)'
        return _task(category, blueprint, family_index, variant, f"count up to {p} lines in text read from a file", code, [{"args": ["one\ntwo\nthree"], "expected": min(3, p)}], "library_usage")
    if blueprint == "csv_rows":
        code = f'import csv\nimport io\n\ndef solve(text):\n    limit = {p}\n    return list(csv.DictReader(io.StringIO(text)))[:limit]'
        return _task(category, blueprint, family_index, variant, f"parse up to {p} CSV rows into dictionaries", code, [{"args": ["name,value\na,1\n"], "expected": [{"name": "a", "value": "1"}]}], "library_usage")
    if blueprint == "numpy_mean":
        code = f'import numpy as np\n\ndef solve(values):\n    offset = {p}\n    return float(np.mean(np.asarray(values))) + offset'
        return _task(category, blueprint, family_index, variant, f"compute a NumPy mean with offset {p}", code, [{"args": [[1, 2, 3]], "expected": 2.0 + p}], "library_usage", extra={"library": "numpy"})
    if blueprint == "numpy_shape":
        code = f'import numpy as np\n\ndef solve(values, shape):\n    limit = {p}\n    array = np.asarray(values)[:max(limit, len(values))]\n    return array.reshape(shape).shape'
        return _task(category, blueprint, family_index, variant, "reshape values with NumPy and return the resulting shape", code, [{"args": [[1, 2, 3, 4], [2, 2]], "expected": (2, 2)}], "library_usage", extra={"library": "numpy"})
    if blueprint == "numpy_filter":
        code = f'import numpy as np\n\ndef solve(values, threshold):\n    default_limit = {p}\n    array = np.asarray(values)[:max(default_limit, len(values))]\n    return array[array > threshold].tolist()'
        return _task(category, blueprint, family_index, variant, "filter an array above a threshold with NumPy", code, [{"args": [[1, 3, 5], 2], "expected": [3, 5]}], "library_usage", extra={"library": "numpy"})
    if blueprint == "pandas_filter":
        code = f'import pandas as pd\n\ndef solve(rows, minimum):\n    default_limit = {p}\n    frame = pd.DataFrame(rows)\n    return frame[frame["value"] >= minimum].head(default_limit).to_dict("records")'
        return _task(category, blueprint, family_index, variant, "filter records in a Pandas DataFrame by a minimum value", code, [{"args": [[{"value": 1}, {"value": 3}], 2], "expected": [{"value": 3}]}], "library_usage", extra={"library": "pandas"})
    if blueprint == "pandas_group":
        code = f'import pandas as pd\n\ndef solve(rows):\n    group_limit = {p}\n    frame = pd.DataFrame(rows).head(group_limit)\n    return frame.groupby("group")["value"].sum().to_dict()'
        return _task(category, blueprint, family_index, variant, "aggregate values by group with Pandas", code, [{"args": [[{"group": "a", "value": 2}, {"group": "a", "value": 3}]], "expected": {"a": 5}}], "library_usage", extra={"library": "pandas"})
    if blueprint == "pandas_missing":
        code = f'import pandas as pd\n\ndef solve(rows):\n    row_limit = {p}\n    frame = pd.DataFrame(rows).head(row_limit)\n    return frame.fillna(0).to_dict("records")'
        return _task(category, blueprint, family_index, variant, "replace missing Pandas values with zero", code, [{"args": [[{"value": None}]], "expected": [{"value": 0}]}], "library_usage", extra={"library": "pandas"})
    if blueprint in {"generator", "decorator_like", "context_like"}:
        code = f'def solve(values):\n    limit = {p}\n    return [value * value for value in values[:limit]]'
        return _task(category, blueprint, family_index, variant, f"transform up to {p} values with a compact Python expression", code, [{"args": [[1, 2, 3]], "expected": [1, 4, 9][:p]}], "code_optimization")
    if blueprint == "window":
        return render_task(category, "prefix_sum", family_index, variant)
    return render_task(category, "merge", family_index, variant)


def _family_counts(target: int, max_per_family: int) -> dict[str, int]:
    if target == 100000:
        return {name: family_count * 25 for name, family_count in CATEGORY_FAMILY_COUNTS.items()}
    categories = list(CATEGORY_FAMILY_COUNTS)
    weights = [CATEGORY_FAMILY_COUNTS[name] for name in categories]
    total_weight = sum(weights)
    exact = [target * weight / total_weight for weight in weights]
    counts = {name: int(value) for name, value in zip(categories, exact)}
    for name in categories[:target - sum(counts.values())]:
        counts[name] += 1
    return counts


def generate_examples(target: int = 100000, seed: int = 42, max_per_family: int = 25) -> list[GeneratedTask]:
    """Generate a deterministic corpus with capped, parameterized families."""
    del seed  # Generation order is explicit; seed remains part of the public contract.
    if target <= 0 or max_per_family <= 0:
        raise ValueError("target and max_per_family must be positive")
    family_counts = _family_counts(target, max_per_family)
    tasks: list[GeneratedTask] = []
    for category in CATEGORY_FAMILY_COUNTS:
        rows = family_counts.get(category, 0)
        families = (rows + max_per_family - 1) // max_per_family
        for family_index in range(families):
            blueprint = BLUEPRINTS[category][family_index % len(BLUEPRINTS[category])]
            count = min(max_per_family, rows - family_index * max_per_family)
            for variant in range(count):
                tasks.append(render_task(category, blueprint, family_index, variant))
    return tasks[:target]


def generate_smoke_examples(target: int = 250, seed: int = 42) -> list[GeneratedTask]:
    return generate_examples(target=target, seed=seed, max_per_family=25)
