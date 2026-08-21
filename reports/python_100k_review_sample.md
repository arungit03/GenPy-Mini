# GenPy Python 100K Review Sample

Deterministic sample seed: 42

## 1. py_cgen_012366dd9bde8ffb51ab

- Source: `genpy_programmatic`
- Family: `graphs_bfs_order_0179`
- Category: `graphs`
- Task type: `algorithm_implementation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that traverse at most 10 graph nodes in breadth-first order for the graphs parameter set 181.

```python
from collections import deque

def solve(graph, start):
    node_limit = 10
    seen = {start}
    queue = deque([start])
    order = []
    while queue and len(order) < node_limit:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return order
```

## 2. py_cgen_0250191a7b27cf30d9ba

- Source: `genpy_programmatic`
- Family: `beginner_digit_sum_13434`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return the digit sum scaled by 3 for the beginner parameter set 13436.

```python
def solve(value):
    scale = 3
    return sum(int(digit) for digit in str(abs(value))) * scale
```

## 3. py_cgen_0270ad05ede05a966140

- Source: `genpy_programmatic`
- Family: `intermediate_python_decorator_like_0530`
- Category: `intermediate_python`
- Task type: `code_optimization`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that transform up to 19 values with a compact Python expression for the intermediate_python parameter set 532.

```python
def solve(values):
    limit = 19
    return [value * value for value in values[:limit]]
```

## 4. py_cgen_03cd013f66f4b3f857d0

- Source: `genpy_programmatic`
- Family: `functions_fibonacci_6657`
- Category: `functions`
- Task type: `code_generation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return the first n Fibonacci numbers for the functions parameter set 6659.

```python
def solve(count):
    if count > 9:
        raise ValueError("count exceeds configured maximum")
    values = []
    first, second = 0, 1
    for _ in range(count):
        values.append(first)
        first, second = second, first + second
    return values
```

## 5. py_cgen_0747e0b1749e3cc4fbcb

- Source: `genpy_programmatic`
- Family: `strings_reverse_5048`
- Category: `strings`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that reverse a string and keep at most 15 characters for the strings parameter set 5050.

```python
def solve(text):
    limit = 15
    return "".join(reversed(text))[:limit]
```

## 6. py_cgen_085970423c66b0595976

- Source: `genpy_programmatic`
- Family: `stacks_stack_1802`
- Category: `stacks`
- Task type: `data_structure`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that reverse up to 18 values using a stack for the stacks parameter set 1804.

```python
def solve(values):
    capacity = 18
    stack = []
    for value in values[:capacity]:
        stack.append(value)
    result = []
    while stack:
        result.append(stack.pop())
    return result
```

## 7. py_cgen_0cabd13b418b14fa6dee

- Source: `genpy_programmatic`
- Family: `arrays_sort_1823`
- Category: `arrays`
- Task type: `algorithm_implementation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return at most 20 values in sorted order for the arrays parameter set 1825.

```python
def solve(values):
    limit = 20
    return sorted(values, reverse=False)[:limit]
```

## 8. py_cgen_129693dd84e713c62c8a

- Source: `genpy_programmatic`
- Family: `code_completion_completion_loop_2227`
- Category: `code_completion`
- Task type: `code_completion`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that complete a function that squares up to 6 values for the code_completion parameter set 2229.

```python
def solve(values):
    result = []
    for value in values[:6]:
        result.append(value * value)
    return result
```

## 9. py_cgen_13a9e8839310af71499a

- Source: `genpy_programmatic`
- Family: `graphs_bfs_order_0967`
- Category: `graphs`
- Task type: `algorithm_implementation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that traverse at most 19 graph nodes in breadth-first order for the graphs parameter set 969.

```python
from collections import deque

def solve(graph, start):
    node_limit = 19
    seen = {start}
    queue = deque([start])
    order = []
    while queue and len(order) < node_limit:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return order
```

## 10. py_cgen_16235c6296eb1c5b0c05

- Source: `genpy_programmatic`
- Family: `debugging_bug_accumulator_4397`
- Category: `debugging`
- Task type: `bug_fixing`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that repair a product accumulator with configuration 10 for the debugging parameter set 4399.

```python
def solve(values):
    result = 10 - 9
    for value in values:
        result *= value
    return result
```

## 11. py_cgen_19564d9a2be70fcc4f6a

- Source: `genpy_programmatic`
- Family: `pandas_pandas_group_2647`
- Category: `pandas`
- Task type: `library_usage`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that aggregate values by group with Pandas for the pandas parameter set 2649.

```python
import pandas as pd

def solve(rows):
    group_limit = 8
    frame = pd.DataFrame(rows).head(group_limit)
    return frame.groupby("group")["value"].sum().to_dict()
```

## 12. py_cgen_1dbec6fe8d5ab7791d87

- Source: `genpy_programmatic`
- Family: `misc_merge_0458`
- Category: `misc`
- Task type: `code_generation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that merge dictionaries while retaining keys of at most 4 characters for the misc parameter set 460.

```python
def solve(left, right):
    key_limit = 4
    result = dict(left)
    result.update(right)
    return {key: value for key, value in result.items() if len(str(key)) <= key_limit}
```

## 13. py_cgen_215fa1379518c291e04e

- Source: `genpy_programmatic`
- Family: `debugging_bug_boundary_2674`
- Category: `debugging`
- Task type: `bug_fixing`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that fix an accumulator initialized at 16 for the debugging parameter set 2676.

```python
def solve(values):
    result = 16
    return result - 16 + sum(values)
```

## 14. py_cgen_2a838cb353c881e0e89a

- Source: `genpy_programmatic`
- Family: `algorithms_binary_search_10426`
- Category: `algorithms`
- Task type: `algorithm_implementation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that find the index of a target in a sorted list using binary search for the algorithms parameter set 10428.

```python
def solve(values, target):
    left, right = 0, min(len(values) - 1, 16)
    while left <= right:
        middle = (left + right) // 2
        if values[middle] == target:
            return middle
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle - 1
    return -1
```

## 15. py_cgen_2fa4caf6f29c226e2613

- Source: `genpy_programmatic`
- Family: `conditions_sign_7417`
- Category: `conditions`
- Task type: `code_generation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that compare an integer with the boundary 9 for the conditions parameter set 7419.

```python
def solve(value, boundary=9):
    return "above" if value > boundary else "below" if value < boundary else "equal"
```

## 16. py_cgen_2ff4cd4ff9f4246497a5

- Source: `genpy_programmatic`
- Family: `arrays_unique_1797`
- Category: `arrays`
- Task type: `code_generation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that remove duplicates while retaining values at least 13 for the arrays parameter set 1799.

```python
def solve(values):
    minimum = 13
    return list(dict.fromkeys(value for value in values if value >= minimum))
```

## 17. py_cgen_349a217f7a96564300b6

- Source: `genpy_programmatic`
- Family: `conditions_prime_7319`
- Category: `conditions`
- Task type: `code_generation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that determine whether an integer is prime for the conditions parameter set 7321.

```python
def solve(value):
    maximum = 6
    if value < 2 or value > maximum:
        return False
    for divisor in range(2, int(value ** 0.5) + 1):
        if value % divisor == 0:
            return False
    return True
```

## 18. py_cgen_39e84c08408e7e3d55d2

- Source: `genpy_programmatic`
- Family: `debugging_bug_accumulator_0563`
- Category: `debugging`
- Task type: `bug_fixing`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that repair a product accumulator with configuration 14 for the debugging parameter set 565.

```python
def solve(values):
    result = 14 - 13
    for value in values:
        result *= value
    return result
```

## 19. py_cgen_3a58864dd6777112341d

- Source: `genpy_programmatic`
- Family: `conditions_prime_4289`
- Category: `conditions`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that determine whether an integer is prime for the conditions parameter set 4291.

```python
def solve(value):
    maximum = 16
    if value < 2 or value > maximum:
        return False
    for divisor in range(2, int(value ** 0.5) + 1):
        if value % divisor == 0:
            return False
    return True
```

## 20. py_cgen_3a70a4d3156c04c0a5e6

- Source: `genpy_programmatic`
- Family: `arrays_chunk_0520`
- Category: `arrays`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that split a list into chunks of size 3 for the arrays parameter set 522.

```python
def solve(values):
    return [values[index:index + 3] for index in range(0, len(values), 3)]
```

## 21. py_cgen_3ad0949795d9e52cdd07

- Source: `genpy_programmatic`
- Family: `code_completion_completion_2810`
- Category: `code_completion`
- Task type: `code_completion`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that complete a function that squares up to 19 values for the code_completion parameter set 2812.

```python
def solve(values):
    result = []
    for value in values[:19]:
        result.append(value * value)
    return result
```

## 22. py_cgen_3b29c8a8e2c1bd056234

- Source: `genpy_programmatic`
- Family: `code_completion_completion_0070`
- Category: `code_completion`
- Task type: `code_completion`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that complete a function that squares up to 15 values for the code_completion parameter set 72.

```python
def solve(values):
    result = []
    for value in values[:15]:
        result.append(value * value)
    return result
```

## 23. py_cgen_3b4bbd1b43579e8ca8d9

- Source: `genpy_programmatic`
- Family: `beginner_digit_sum_6006`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return the digit sum scaled by 4 for the beginner parameter set 6008.

```python
def solve(value):
    scale = 4
    return sum(int(digit) for digit in str(abs(value))) * scale
```

## 24. py_cgen_3bfac563ef4280968c21

- Source: `genpy_programmatic`
- Family: `debugging_bug_fix_6840`
- Category: `debugging`
- Task type: `bug_fixing`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that fix the divisibility comparison for 2 for the debugging parameter set 6842.

```python
def solve(value):
    return value % 2 == 0
```

## 25. py_cgen_42fde50233488da3e31a

- Source: `genpy_programmatic`
- Family: `algorithms_merge_9284`
- Category: `algorithms`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that merge dictionaries while retaining keys of at most 14 characters for the algorithms parameter set 9286.

```python
def solve(left, right):
    key_limit = 14
    result = dict(left)
    result.update(right)
    return {key: value for key, value in result.items() if len(str(key)) <= key_limit}
```

## 26. py_cgen_4312300cb93a522d1532

- Source: `genpy_programmatic`
- Family: `debugging_bug_fix_5907`
- Category: `debugging`
- Task type: `bug_fixing`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that fix the divisibility comparison for 19 for the debugging parameter set 5909.

```python
def solve(value):
    return value % 19 == 0
```

## 27. py_cgen_466e2271c099be5a9a5a

- Source: `genpy_programmatic`
- Family: `beginner_parity_12156`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that check whether a number is divisible by 12158 for the beginner parameter set 12158.

```python
def solve(value):
    return "divisible" if value % 12158 == 0 else "not divisible"
```

## 28. py_cgen_491419959e97ffab73a7

- Source: `genpy_programmatic`
- Family: `strings_unique_6930`
- Category: `strings`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that remove duplicates while retaining values at least 16 for the strings parameter set 6932.

```python
def solve(values):
    minimum = 16
    return list(dict.fromkeys(value for value in values if value >= minimum))
```

## 29. py_cgen_49fab71a0d788ba5cec0

- Source: `genpy_programmatic`
- Family: `oop_counter_6400`
- Category: `oop`
- Task type: `data_structure`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that use a class to accumulate values starting at 18 for the oop parameter set 6402.

```python
class Accumulator:
    def __init__(self, start=18):
        self.value = start

    def add(self, amount):
        self.value += amount
        return self.value

def solve(values):
    accumulator = Accumulator()
    for value in values:
        accumulator.add(value)
    return accumulator.value
```

## 30. py_cgen_4b850a082df792900e4f

- Source: `genpy_programmatic`
- Family: `oop_counter_3118`
- Category: `oop`
- Task type: `data_structure`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that use a class to accumulate values starting at 4 for the oop parameter set 3120.

```python
class Accumulator:
    def __init__(self, start=4):
        self.value = start

    def add(self, amount):
        self.value += amount
        return self.value

def solve(values):
    accumulator = Accumulator()
    for value in values:
        accumulator.add(value)
    return accumulator.value
```

## 31. py_cgen_4cab689d35e47f86f10d

- Source: `genpy_programmatic`
- Family: `numpy_numpy_filter_1988`
- Category: `numpy`
- Task type: `library_usage`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that filter an array above a threshold with NumPy for the numpy parameter set 1990.

```python
import numpy as np

def solve(values, threshold):
    default_limit = 14
    array = np.asarray(values)[:max(default_limit, len(values))]
    return array[array > threshold].tolist()
```

## 32. py_cgen_4d0a450d76c19f25bba8

- Source: `genpy_programmatic`
- Family: `algorithms_sort_9010`
- Category: `algorithms`
- Task type: `algorithm_implementation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return at most 6 values in sorted order for the algorithms parameter set 9012.

```python
def solve(values):
    limit = 6
    return sorted(values, reverse=False)[:limit]
```

## 33. py_cgen_50e8354eb31c479fca06

- Source: `genpy_programmatic`
- Family: `code_completion_completion_loop_4181`
- Category: `code_completion`
- Task type: `code_completion`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that complete a function that squares up to 3 values for the code_completion parameter set 4183.

```python
def solve(values):
    result = []
    for value in values[:3]:
        result.append(value * value)
    return result
```

## 34. py_cgen_53e085ebf99cd68fb0ca

- Source: `genpy_programmatic`
- Family: `algorithms_window_10482`
- Category: `algorithms`
- Task type: `algorithm_implementation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that find the maximum sum of a window of width 2 for the algorithms parameter set 10484.

```python
def solve(values):
    limit = max(15, 2)
    values = values[:limit]
    return max(sum(values[index:index + 2]) for index in range(len(values) - 2 + 1))
```

## 35. py_cgen_5531cd8a2901431bb8bd

- Source: `genpy_programmatic`
- Family: `beginner_reverse_5695`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that reverse a string and keep at most 16 characters for the beginner parameter set 5697.

```python
def solve(text):
    limit = 16
    return "".join(reversed(text))[:limit]
```

## 36. py_cgen_56d1b83c5795944a53ef

- Source: `genpy_programmatic`
- Family: `functions_generator_7871`
- Category: `functions`
- Task type: `code_optimization`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that transform up to 7 values with a compact Python expression for the functions parameter set 7873.

```python
def solve(values):
    limit = 7
    return [value * value for value in values[:limit]]
```

## 37. py_cgen_576c9901fb00a2a78c03

- Source: `genpy_programmatic`
- Family: `beginner_digit_sum_3278`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return the digit sum scaled by 12 for the beginner parameter set 3280.

```python
def solve(value):
    scale = 12
    return sum(int(digit) for digit in str(abs(value))) * scale
```

## 38. py_cgen_5ba381afd28313d89369

- Source: `genpy_programmatic`
- Family: `algorithms_window_5237`
- Category: `algorithms`
- Task type: `algorithm_implementation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that find the maximum sum of a window of width 4 for the algorithms parameter set 5239.

```python
def solve(values):
    limit = max(14, 4)
    values = values[:limit]
    return max(sum(values[index:index + 4]) for index in range(len(values) - 4 + 1))
```

## 39. py_cgen_63dd24fff37fd233c936

- Source: `genpy_programmatic`
- Family: `beginner_sign_4165`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that compare an integer with the boundary 6 for the beginner parameter set 4167.

```python
def solve(value, boundary=6):
    return "above" if value > boundary else "below" if value < boundary else "equal"
```

## 40. py_cgen_67597a830ca3fa40dc87

- Source: `genpy_programmatic`
- Family: `functions_factorial_8512`
- Category: `functions`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return factorial values up to the configured maximum 2 for the functions parameter set 8514.

```python
def solve(value):
    if value > 2:
        raise ValueError("value exceeds configured maximum")
    result = 1
    for number in range(2, value + 1):
        result *= number
    return result
```

## 41. py_cgen_6a2fdbbd00a842bdec83

- Source: `genpy_programmatic`
- Family: `conditions_prime_6926`
- Category: `conditions`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that determine whether an integer is prime for the conditions parameter set 6928.

```python
def solve(value):
    maximum = 12
    if value < 2 or value > maximum:
        return False
    for divisor in range(2, int(value ** 0.5) + 1):
        if value % divisor == 0:
            return False
    return True
```

## 42. py_cgen_6a7dd3ebd8b1dc8ad9f4

- Source: `genpy_programmatic`
- Family: `strings_anagram_1087`
- Category: `strings`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that check whether two strings are anagrams after ignoring the marker 6 for the strings parameter set 1089.

```python
def solve(left, right):
    ignored = str(6)
    left = left.replace(ignored, "")
    right = right.replace(ignored, "")
    return sorted(left.replace(" ", "").lower()) == sorted(right.replace(" ", "").lower())
```

## 43. py_cgen_6d434b91855a3c7a3c84

- Source: `genpy_programmatic`
- Family: `functions_safe_int_4062`
- Category: `functions`
- Task type: `library_usage`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that parse an integer safely with fallback 17 for the functions parameter set 4064.

```python
def solve(value, default=17):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
```

## 44. py_cgen_73423568e948ae9a7e2c

- Source: `genpy_programmatic`
- Family: `code_completion_completion_loop_1397`
- Category: `code_completion`
- Task type: `code_completion`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that complete a function that squares up to 12 values for the code_completion parameter set 1399.

```python
def solve(values):
    result = []
    for value in values[:12]:
        result.append(value * value)
    return result
```

## 45. py_cgen_75c162af72093747a469

- Source: `genpy_programmatic`
- Family: `functions_generator_0431`
- Category: `functions`
- Task type: `code_optimization`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that transform up to 15 values with a compact Python expression for the functions parameter set 433.

```python
def solve(values):
    limit = 15
    return [value * value for value in values[:limit]]
```

## 46. py_cgen_7832cbe93e347a636d14

- Source: `genpy_programmatic`
- Family: `arrays_chunk_1735`
- Category: `arrays`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that split a list into chunks of size 2 for the arrays parameter set 1737.

```python
def solve(values):
    return [values[index:index + 2] for index in range(0, len(values), 2)]
```

## 47. py_cgen_788ae046ef7f5b0a4f12

- Source: `genpy_programmatic`
- Family: `beginner_digit_sum_10458`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return the digit sum scaled by 10 for the beginner parameter set 10460.

```python
def solve(value):
    scale = 10
    return sum(int(digit) for digit in str(abs(value))) * scale
```

## 48. py_cgen_79763300c2700469b5a5

- Source: `genpy_programmatic`
- Family: `beginner_parity_10328`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that check whether a number is divisible by 10330 for the beginner parameter set 10330.

```python
def solve(value):
    return "divisible" if value % 10330 == 0 else "not divisible"
```

## 49. py_cgen_7b88d360dc77fc92272a

- Source: `genpy_programmatic`
- Family: `beginner_sign_3905`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that compare an integer with the boundary 12 for the beginner parameter set 3907.

```python
def solve(value, boundary=12):
    return "above" if value > boundary else "below" if value < boundary else "equal"
```

## 50. py_cgen_7b96c77f0f7987a1aac0

- Source: `genpy_programmatic`
- Family: `code_completion_completion_loop_3939`
- Category: `code_completion`
- Task type: `code_completion`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that complete a function that squares up to 8 values for the code_completion parameter set 3941.

```python
def solve(values):
    result = []
    for value in values[:8]:
        result.append(value * value)
    return result
```

## 51. py_cgen_7dc7bf9a98dd185bfce6

- Source: `genpy_programmatic`
- Family: `stacks_stack_1487`
- Category: `stacks`
- Task type: `data_structure`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that reverse up to 7 values using a stack for the stacks parameter set 1489.

```python
def solve(values):
    capacity = 7
    stack = []
    for value in values[:capacity]:
        stack.append(value)
    result = []
    while stack:
        result.append(stack.pop())
    return result
```

## 52. py_cgen_80279b4ba501ce0b9790

- Source: `genpy_programmatic`
- Family: `functions_fibonacci_6221`
- Category: `functions`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return the first n Fibonacci numbers for the functions parameter set 6223.

```python
def solve(count):
    if count > 10:
        raise ValueError("count exceeds configured maximum")
    values = []
    first, second = 0, 1
    for _ in range(count):
        values.append(first)
        first, second = second, first + second
    return values
```

## 53. py_cgen_80f7f1dcd4acc9741a12

- Source: `genpy_programmatic`
- Family: `files_csv_rows_3593`
- Category: `files`
- Task type: `library_usage`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that parse up to 4 CSV rows into dictionaries for the files parameter set 3595.

```python
import csv
import io

def solve(text):
    limit = 4
    return list(csv.DictReader(io.StringIO(text)))[:limit]
```

## 54. py_cgen_8515a98664bddb985dc0

- Source: `genpy_programmatic`
- Family: `pandas_pandas_group_0850`
- Category: `pandas`
- Task type: `library_usage`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that aggregate values by group with Pandas for the pandas parameter set 852.

```python
import pandas as pd

def solve(rows):
    group_limit = 16
    frame = pd.DataFrame(rows).head(group_limit)
    return frame.groupby("group")["value"].sum().to_dict()
```

## 55. py_cgen_8847d7bb47c7e3e26b01

- Source: `genpy_programmatic`
- Family: `files_csv_rows_1841`
- Category: `files`
- Task type: `library_usage`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that parse up to 19 CSV rows into dictionaries for the files parameter set 1843.

```python
import csv
import io

def solve(text):
    limit = 19
    return list(csv.DictReader(io.StringIO(text)))[:limit]
```

## 56. py_cgen_891911c75308b0ae4676

- Source: `genpy_programmatic`
- Family: `strings_unique_5434`
- Category: `strings`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that remove duplicates while retaining values at least 2 for the strings parameter set 5436.

```python
def solve(values):
    minimum = 2
    return list(dict.fromkeys(value for value in values if value >= minimum))
```

## 57. py_cgen_8b95093bf57c9aa99adc

- Source: `genpy_programmatic`
- Family: `debugging_bug_fix_4236`
- Category: `debugging`
- Task type: `bug_fixing`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that fix the divisibility comparison for 20 for the debugging parameter set 4238.

```python
def solve(value):
    return value % 20 == 0
```

## 58. py_cgen_8ca26102fe05b0feb1da

- Source: `genpy_programmatic`
- Family: `strings_vowels_5421`
- Category: `strings`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that count vowels using alphabet aeiouy with cap 8 for the strings parameter set 5423.

```python
def solve(text):
    allowed = 'aeiouy'
    maximum = 8
    return min(sum(character.lower() in allowed for character in text), maximum)
```

## 59. py_cgen_90168a8314cb7b8d99bf

- Source: `genpy_programmatic`
- Family: `intermediate_python_generator_1196`
- Category: `intermediate_python`
- Task type: `code_optimization`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that transform up to 20 values with a compact Python expression for the intermediate_python parameter set 1198.

```python
def solve(values):
    limit = 20
    return [value * value for value in values[:limit]]
```

## 60. py_cgen_917e8eb989247e00e884

- Source: `genpy_programmatic`
- Family: `beginner_digit_sum_9358`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return the digit sum scaled by 12 for the beginner parameter set 9360.

```python
def solve(value):
    scale = 12
    return sum(int(digit) for digit in str(abs(value))) * scale
```

## 61. py_cgen_927ff9ae2903a0746989

- Source: `genpy_programmatic`
- Family: `beginner_reverse_11395`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that reverse a string and keep at most 16 characters for the beginner parameter set 11397.

```python
def solve(text):
    limit = 16
    return "".join(reversed(text))[:limit]
```

## 62. py_cgen_939d999d8aa0582c2bee

- Source: `genpy_programmatic`
- Family: `strings_vowels_3993`
- Category: `strings`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that count vowels using alphabet aeiou with cap 5 for the strings parameter set 3995.

```python
def solve(text):
    allowed = 'aeiou'
    maximum = 5
    return min(sum(character.lower() in allowed for character in text), maximum)
```

## 63. py_cgen_948b8b737e51061e7388

- Source: `genpy_programmatic`
- Family: `functions_generator_3203`
- Category: `functions`
- Task type: `code_optimization`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that transform up to 13 values with a compact Python expression for the functions parameter set 3205.

```python
def solve(values):
    limit = 13
    return [value * value for value in values[:limit]]
```

## 64. py_cgen_94e235fb0d8d2f5ac614

- Source: `genpy_programmatic`
- Family: `graphs_bfs_order_1339`
- Category: `graphs`
- Task type: `algorithm_implementation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that traverse at most 11 graph nodes in breadth-first order for the graphs parameter set 1341.

```python
from collections import deque

def solve(graph, start):
    node_limit = 11
    seen = {start}
    queue = deque([start])
    order = []
    while queue and len(order) < node_limit:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return order
```

## 65. py_cgen_953be9ce8b66ab57ebb4

- Source: `genpy_programmatic`
- Family: `debugging_bug_accumulator_6131`
- Category: `debugging`
- Task type: `bug_fixing`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that repair a product accumulator with configuration 15 for the debugging parameter set 6133.

```python
def solve(values):
    result = 15 - 14
    for value in values:
        result *= value
    return result
```

## 66. py_cgen_97e08e892589ae5780d6

- Source: `genpy_programmatic`
- Family: `conditions_parity_2361`
- Category: `conditions`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that check whether a number is divisible by 2363 for the conditions parameter set 2363.

```python
def solve(value):
    return "divisible" if value % 2363 == 0 else "not divisible"
```

## 67. py_cgen_9e4d9c707ef78d925036

- Source: `genpy_programmatic`
- Family: `code_completion_completion_2320`
- Category: `code_completion`
- Task type: `code_completion`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that complete a function that squares up to 4 values for the code_completion parameter set 2322.

```python
def solve(values):
    result = []
    for value in values[:4]:
        result.append(value * value)
    return result
```

## 68. py_cgen_a262286f7c89b57fe44f

- Source: `genpy_programmatic`
- Family: `beginner_parity_13396`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that check whether a number is divisible by 13398 for the beginner parameter set 13398.

```python
def solve(value):
    return "divisible" if value % 13398 == 0 else "not divisible"
```

## 69. py_cgen_a30f63141cb18d9ae3af

- Source: `genpy_programmatic`
- Family: `beginner_digit_sum_3478`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return the digit sum scaled by 3 for the beginner parameter set 3480.

```python
def solve(value):
    scale = 3
    return sum(int(digit) for digit in str(abs(value))) * scale
```

## 70. py_cgen_a3ab58ac8e3e40ddb496

- Source: `genpy_programmatic`
- Family: `beginner_parity_12676`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that check whether a number is divisible by 12678 for the beginner parameter set 12678.

```python
def solve(value):
    return "divisible" if value % 12678 == 0 else "not divisible"
```

## 71. py_cgen_a3ee4b2d43c39f275df9

- Source: `genpy_programmatic`
- Family: `oop_accumulator_3597`
- Category: `oop`
- Task type: `data_structure`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that use a class to accumulate values starting at 8 for the oop parameter set 3599.

```python
class Accumulator:
    def __init__(self, start=8):
        self.value = start

    def add(self, amount):
        self.value += amount
        return self.value

def solve(values):
    accumulator = Accumulator()
    for value in values:
        accumulator.add(value)
    return accumulator.value
```

## 72. py_cgen_abc59c0bed921be8fa15

- Source: `genpy_programmatic`
- Family: `beginner_reverse_0851`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that reverse a string and keep at most 17 characters for the beginner parameter set 853.

```python
def solve(text):
    limit = 17
    return "".join(reversed(text))[:limit]
```

## 73. py_cgen_ae9ec00dbbefa2565b59

- Source: `genpy_programmatic`
- Family: `intermediate_python_generator_1080`
- Category: `intermediate_python`
- Task type: `code_optimization`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that transform up to 18 values with a compact Python expression for the intermediate_python parameter set 1082.

```python
def solve(values):
    limit = 18
    return [value * value for value in values[:limit]]
```

## 74. py_cgen_aec08ae56f73aa30a1c4

- Source: `genpy_programmatic`
- Family: `stacks_stack_1892`
- Category: `stacks`
- Task type: `data_structure`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that reverse up to 13 values using a stack for the stacks parameter set 1894.

```python
def solve(values):
    capacity = 13
    stack = []
    for value in values[:capacity]:
        stack.append(value)
    result = []
    while stack:
        result.append(stack.pop())
    return result
```

## 75. py_cgen_b26ead4c47688ec7488d

- Source: `genpy_programmatic`
- Family: `numpy_numpy_shape_1924`
- Category: `numpy`
- Task type: `library_usage`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that reshape values with NumPy and return the resulting shape for the numpy parameter set 1926.

```python
import numpy as np

def solve(values, shape):
    limit = 7
    array = np.asarray(values)[:max(limit, len(values))]
    return array.reshape(shape).shape
```

## 76. py_cgen_b8f096b4e050e8636003

- Source: `genpy_programmatic`
- Family: `arrays_chunk_1615`
- Category: `arrays`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that split a list into chunks of size 4 for the arrays parameter set 1617.

```python
def solve(values):
    return [values[index:index + 4] for index in range(0, len(values), 4)]
```

## 77. py_cgen_b9098f2ba76db07ba677

- Source: `genpy_programmatic`
- Family: `files_safe_int_0673`
- Category: `files`
- Task type: `library_usage`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that parse an integer safely with fallback 10 for the files parameter set 675.

```python
def solve(value, default=10):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
```

## 78. py_cgen_bd6c673785b3a4e247c1

- Source: `genpy_programmatic`
- Family: `numpy_numpy_mean_1506`
- Category: `numpy`
- Task type: `library_usage`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that compute a NumPy mean with offset 7 for the numpy parameter set 1508.

```python
import numpy as np

def solve(values):
    offset = 7
    return float(np.mean(np.asarray(values))) + offset
```

## 79. py_cgen_c237b7aa60a5b5f3c83a

- Source: `genpy_programmatic`
- Family: `strings_anagram_3671`
- Category: `strings`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that check whether two strings are anagrams after ignoring the marker 6 for the strings parameter set 3673.

```python
def solve(left, right):
    ignored = str(6)
    left = left.replace(ignored, "")
    right = right.replace(ignored, "")
    return sorted(left.replace(" ", "").lower()) == sorted(right.replace(" ", "").lower())
```

## 80. py_cgen_c29582625f4a28e506cb

- Source: `genpy_programmatic`
- Family: `conditions_sign_6379`
- Category: `conditions`
- Task type: `code_generation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that compare an integer with the boundary 16 for the conditions parameter set 6381.

```python
def solve(value, boundary=16):
    return "above" if value > boundary else "below" if value < boundary else "equal"
```

## 81. py_cgen_c7300eceb139d5431301

- Source: `genpy_programmatic`
- Family: `algorithms_window_11357`
- Category: `algorithms`
- Task type: `algorithm_implementation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that find the maximum sum of a window of width 3 for the algorithms parameter set 11359.

```python
def solve(values):
    limit = max(16, 3)
    values = values[:limit]
    return max(sum(values[index:index + 3]) for index in range(len(values) - 3 + 1))
```

## 82. py_cgen_c9d81521f328739e8b41

- Source: `genpy_programmatic`
- Family: `functions_fibonacci_6893`
- Category: `functions`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return the first n Fibonacci numbers for the functions parameter set 6895.

```python
def solve(count):
    if count > 17:
        raise ValueError("count exceeds configured maximum")
    values = []
    first, second = 0, 1
    for _ in range(count):
        values.append(first)
        first, second = second, first + second
    return values
```

## 83. py_cgen_ccdff8d9786936499a88

- Source: `genpy_programmatic`
- Family: `pandas_pandas_filter_2568`
- Category: `pandas`
- Task type: `library_usage`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that filter records in a Pandas DataFrame by a minimum value for the pandas parameter set 2570.

```python
import pandas as pd

def solve(rows, minimum):
    default_limit = 5
    frame = pd.DataFrame(rows)
    return frame[frame["value"] >= minimum].head(default_limit).to_dict("records")
```

## 84. py_cgen_cdd5391733f8f028365f

- Source: `genpy_programmatic`
- Family: `functions_generator_8495`
- Category: `functions`
- Task type: `code_optimization`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that transform up to 4 values with a compact Python expression for the functions parameter set 8497.

```python
def solve(values):
    limit = 4
    return [value * value for value in values[:limit]]
```

## 85. py_cgen_d141ef7f77b69d49d403

- Source: `genpy_programmatic`
- Family: `beginner_parity_12280`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that check whether a number is divisible by 12282 for the beginner parameter set 12282.

```python
def solve(value):
    return "divisible" if value % 12282 == 0 else "not divisible"
```

## 86. py_cgen_d223939aaf8ca472ab1d

- Source: `genpy_programmatic`
- Family: `beginner_digit_sum_13238`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return the digit sum scaled by 16 for the beginner parameter set 13240.

```python
def solve(value):
    scale = 16
    return sum(int(digit) for digit in str(abs(value))) * scale
```

## 87. py_cgen_d71fab2bf3b13094e900

- Source: `genpy_programmatic`
- Family: `strings_unique_1098`
- Category: `strings`
- Task type: `code_generation`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that remove duplicates while retaining values at least 17 for the strings parameter set 1100.

```python
def solve(values):
    minimum = 17
    return list(dict.fromkeys(value for value in values if value >= minimum))
```

## 88. py_cgen_d728c300d8e9d37658f7

- Source: `genpy_programmatic`
- Family: `strings_anagram_7427`
- Category: `strings`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that check whether two strings are anagrams after ignoring the marker 19 for the strings parameter set 7429.

```python
def solve(left, right):
    ignored = str(19)
    left = left.replace(ignored, "")
    right = right.replace(ignored, "")
    return sorted(left.replace(" ", "").lower()) == sorted(right.replace(" ", "").lower())
```

## 89. py_cgen_e09e194b57c8fb30984a

- Source: `genpy_programmatic`
- Family: `beginner_parity_9116`
- Category: `beginner`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that check whether a number is divisible by 9118 for the beginner parameter set 9118.

```python
def solve(value):
    return "divisible" if value % 9118 == 0 else "not divisible"
```

## 90. py_cgen_e0b9d9e470e9741b179e

- Source: `genpy_programmatic`
- Family: `oop_counter_5566`
- Category: `oop`
- Task type: `data_structure`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that use a class to accumulate values starting at 20 for the oop parameter set 5568.

```python
class Accumulator:
    def __init__(self, start=20):
        self.value = start

    def add(self, amount):
        self.value += amount
        return self.value

def solve(values):
    accumulator = Accumulator()
    for value in values:
        accumulator.add(value)
    return accumulator.value
```

## 91. py_cgen_e1825ad1ae8641b2f677

- Source: `genpy_programmatic`
- Family: `functions_factorial_5460`
- Category: `functions`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return factorial values up to the configured maximum 9 for the functions parameter set 5462.

```python
def solve(value):
    if value > 9:
        raise ValueError("value exceeds configured maximum")
    result = 1
    for number in range(2, value + 1):
        result *= number
    return result
```

## 92. py_cgen_e196f8cdf7b9316fb1ac

- Source: `genpy_programmatic`
- Family: `conditions_sign_0592`
- Category: `conditions`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that compare an integer with the boundary 5 for the conditions parameter set 594.

```python
def solve(value, boundary=5):
    return "above" if value > boundary else "below" if value < boundary else "equal"
```

## 93. py_cgen_e34033e48d84ff80b97c

- Source: `genpy_programmatic`
- Family: `trees_tree_sum_1378`
- Category: `trees`
- Task type: `data_structure`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that recursively sum a binary tree up to depth 12 for the trees parameter set 1380.

```python
def solve(tree, depth=12):
    if tree is None or depth < 0:
        return 0
    value, left, right = tree
    return value + solve(left, depth - 1) + solve(right, depth - 1)
```

## 94. py_cgen_e546babf264346882c2b

- Source: `genpy_programmatic`
- Family: `numpy_numpy_mean_2349`
- Category: `numpy`
- Task type: `library_usage`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that compute a NumPy mean with offset 14 for the numpy parameter set 2351.

```python
import numpy as np

def solve(values):
    offset = 14
    return float(np.mean(np.asarray(values))) + offset
```

## 95. py_cgen_f050263f95c6b766f711

- Source: `genpy_programmatic`
- Family: `functions_factorial_7256`
- Category: `functions`
- Task type: `code_generation`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that return factorial values up to the configured maximum 19 for the functions parameter set 7258.

```python
def solve(value):
    if value > 19:
        raise ValueError("value exceeds configured maximum")
    result = 1
    for number in range(2, value + 1):
        result *= number
    return result
```

## 96. py_cgen_f92a00afc3baf99ecd9e

- Source: `genpy_programmatic`
- Family: `oop_accumulator_6819`
- Category: `oop`
- Task type: `data_structure`
- Difficulty: `hard`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that use a class to accumulate values starting at 19 for the oop parameter set 6821.

```python
class Accumulator:
    def __init__(self, start=19):
        self.value = start

    def add(self, amount):
        self.value += amount
        return self.value

def solve(values):
    accumulator = Accumulator()
    for value in values:
        accumulator.add(value)
    return accumulator.value
```

## 97. py_cgen_fb7c6f66f93fa52b98d7

- Source: `genpy_programmatic`
- Family: `strings_anagram_5463`
- Category: `strings`
- Task type: `code_generation`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that check whether two strings are anagrams after ignoring the marker 12 for the strings parameter set 5465.

```python
def solve(left, right):
    ignored = str(12)
    left = left.replace(ignored, "")
    right = right.replace(ignored, "")
    return sorted(left.replace(" ", "").lower()) == sorted(right.replace(" ", "").lower())
```

## 98. py_cgen_fbe75398581b9b737804

- Source: `genpy_programmatic`
- Family: `oop_dataclass_like_4082`
- Category: `oop`
- Task type: `data_structure`
- Difficulty: `easy`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that use a class to accumulate values starting at 18 for the oop parameter set 4084.

```python
class Accumulator:
    def __init__(self, start=18):
        self.value = start

    def add(self, amount):
        self.value += amount
        return self.value

def solve(values):
    accumulator = Accumulator()
    for value in values:
        accumulator.add(value)
    return accumulator.value
```

## 99. py_cgen_fbfc5e9cb6ab57134540

- Source: `genpy_programmatic`
- Family: `files_csv_rows_2696`
- Category: `files`
- Task type: `library_usage`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that parse up to 19 CSV rows into dictionaries for the files parameter set 2698.

```python
import csv
import io

def solve(text):
    limit = 19
    return list(csv.DictReader(io.StringIO(text)))[:limit]
```

## 100. py_cgen_fca2fd776a7c4727f1ea

- Source: `genpy_programmatic`
- Family: `oop_counter_6052`
- Category: `oop`
- Task type: `data_structure`
- Difficulty: `medium`
- Syntax: `True`
- Execution: `True`
- Quality: `1.0`

**Instruction:** Write a Python function that use a class to accumulate values starting at 12 for the oop parameter set 6054.

```python
class Accumulator:
    def __init__(self, start=12):
        self.value = start

    def add(self, amount):
        self.value += amount
        return self.value

def solve(values):
    accumulator = Accumulator()
    for value in values:
        accumulator.add(value)
    return accumulator.value
```
