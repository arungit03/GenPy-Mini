"""Generate the fresh deterministic GenPy Checkpoint 8-v2 pilot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

GENERATOR_VERSION = "genpy-sft-v2-pilot-v1"
CP7_TRAIN_HASH = "17ba25f0154d1ffa04fdd4b91a22123a0770fe6aa76416ba57e4630264cb0b44"
CP7_VALIDATION_HASH = "7ec5fabfb339e0c9986160193da01d30fe6d46035775f81e9777a4ad92731e97"

CATEGORY_SKILLS = {
    "conditionals": ["parity_even", "sign_label", "clamp_interval", "leap_year", "divisible_by", "score_band", "triangle_possible", "boolean_xor", "inside_interval", "shipping_band", "temperature_label", "access_level"],
    "loops_math": ["factorial", "digit_sum", "greatest_common_divisor", "least_common_multiple", "prime_check", "fibonacci_term", "sum_multiples", "digit_count", "collatz_steps", "integer_power"],
    "strings": ["reverse_text", "palindrome_text", "vowel_count", "first_unique_character", "anagram_check", "rotate_text", "compress_runs", "longest_word", "subsequence_check", "word_count", "acronym", "strip_vowels"],
    "sequences": ["maximum_item", "sum_items", "stable_unique", "chunk_sequence", "rotate_sequence", "flatten_one_level", "running_totals", "interleave_sequences", "two_sum_exists", "transpose_rows", "window_maximum", "partition_values"],
    "dictionaries_sets": ["character_histogram", "invert_mapping", "merge_counters", "common_keys", "group_by_parity", "lookup_default", "sorted_items", "square_mapping"],
    "functions_recursion": ["recursive_factorial", "recursive_sum", "recursive_fibonacci", "recursive_flatten", "compose_increment", "make_counter", "memoized_style", "nested_sum", "tree_height"],
    "comprehensions": ["square_transform", "even_filter", "absolute_transform", "length_mapping", "pair_mapping"],
    "oop": ["point_distance", "counter_sequence", "rectangle_area", "stack_sequence", "inventory_total", "bank_sequence"],
    "exceptions": ["safe_integer", "safe_division", "parse_boolean", "mapping_default", "validate_age"],
    "algorithms": ["binary_search", "insertion_sort", "merge_sorted", "merge_intervals", "two_sum_indices", "hamming_distance", "edit_distance", "reachable_graph", "count_inversions", "kth_smallest", "balanced_brackets", "longest_run", "matrix_diagonal", "change_ways"],
    "debugging": ["repair_maximum", "repair_reverse", "repair_parity", "repair_unique", "repair_search", "repair_factorial", "repair_chunks"],
}
CATEGORY_TARGETS = {"conditionals": 1200, "loops_math": 1000, "strings": 1200, "sequences": 1200, "dictionaries_sets": 800, "functions_recursion": 900, "comprehensions": 500, "oop": 600, "exceptions": 500, "algorithms": 1400, "debugging": 700}
CHALLENGE_ONLY = ["roman_value", "version_key", "decode_runs", "base_digits", "matrix_border", "nearest_neighbor", "snake_case", "unique_window", "weekday_cycle", "balanced_partition", "merge_k_lists", "prefix_products", "coordinate_spiral", "token_buckets", "median_stream"]
CHALLENGE_ONLY_KINDS = ["roman", "version", "decode_runs", "base_digits", "border", "nearest", "snake", "unique_window", "reverse", "balanced", "merge_sorted", "prefix_products", "running", "histogram", "median"]
STYLES = ["function_request", "implementation_request", "signature_completion", "constraint_request"]
SLUG = re.compile(r"[^a-z0-9]+")


def fname(skill: str, split: str, index: int) -> str:
    return "solve_{}_{}_{}".format(SLUG.sub("_", skill), split, index)


def cases(kind: str) -> list[dict]:
    table = {
        "parity": [{"args": [2], "expected": True}, {"args": [7], "expected": False}, {"args": [-4], "expected": True}],
        "sign": [{"args": [-2], "expected": "negative"}, {"args": [0], "expected": "zero"}, {"args": [5], "expected": "positive"}],
        "clamp": [{"args": [3, 0, 10], "expected": 3}, {"args": [-2, 0, 10], "expected": 0}, {"args": [12, 0, 10], "expected": 10}],
        "leap": [{"args": [2000], "expected": True}, {"args": [1900], "expected": False}, {"args": [2024], "expected": True}],
        "divisible": [{"args": [12, 3], "expected": True}, {"args": [14, 3], "expected": False}, {"args": [0, 5], "expected": True}],
        "band": [{"args": [95], "expected": "A"}, {"args": [72], "expected": "C"}, {"args": [41], "expected": "F"}],
        "triangle": [{"args": [3, 4, 5], "expected": True}, {"args": [1, 2, 3], "expected": False}, {"args": [5, 5, 5], "expected": True}],
        "xor": [{"args": [True, False], "expected": True}, {"args": [True, True], "expected": False}, {"args": [False, False], "expected": False}],
        "factorial": [{"args": [0], "expected": 1}, {"args": [1], "expected": 1}, {"args": [5], "expected": 120}],
        "digit_sum": [{"args": [0], "expected": 0}, {"args": [1234], "expected": 10}, {"args": [-58], "expected": 13}],
        "gcd": [{"args": [18, 24], "expected": 6}, {"args": [7, 3], "expected": 1}, {"args": [0, 9], "expected": 9}],
        "lcm": [{"args": [4, 6], "expected": 12}, {"args": [5, 7], "expected": 35}, {"args": [0, 4], "expected": 0}],
        "prime": [{"args": [2], "expected": True}, {"args": [1], "expected": False}, {"args": [17], "expected": True}],
        "fib": [{"args": [0], "expected": 0}, {"args": [1], "expected": 1}, {"args": [8], "expected": 21}],
        "sum_multiples": [{"args": [10, 3], "expected": 18}, {"args": [1, 2], "expected": 0}, {"args": [20, 5], "expected": 50}],
        "digits": [{"args": [0], "expected": 1}, {"args": [42], "expected": 2}, {"args": [-987], "expected": 3}],
        "collatz": [{"args": [1], "expected": 0}, {"args": [2], "expected": 1}, {"args": [6], "expected": 8}],
        "power": [{"args": [2, 0], "expected": 1}, {"args": [3, 2], "expected": 9}, {"args": [5, 3], "expected": 125}],
        "reverse": [{"args": ["hello"], "expected": "olleh"}, {"args": [""], "expected": ""}, {"args": ["GenPy"], "expected": "yPneG"}],
        "palindrome": [{"args": ["level"], "expected": True}, {"args": ["python"], "expected": False}, {"args": [""], "expected": True}],
        "vowels": [{"args": ["Umbrella"], "expected": 3}, {"args": ["rhythm"], "expected": 0}, {"args": ["AEIOU"], "expected": 5}],
        "first_unique": [{"args": ["swiss"], "expected": "w"}, {"args": ["aabb"], "expected": None}, {"args": ["z"], "expected": "z"}],
        "anagram": [{"args": ["listen", "silent"], "expected": True}, {"args": ["cat", "car"], "expected": False}, {"args": ["", ""], "expected": True}],
        "rotate_text": [{"args": ["abcd", 1], "expected": "dabc"}, {"args": ["abcd", 5], "expected": "dabc"}, {"args": ["", 2], "expected": ""}],
        "compress": [{"args": ["aaabbc"], "expected": "a3b2c1"}, {"args": ["x"], "expected": "x1"}, {"args": [""], "expected": ""}],
        "longest_word": [{"args": ["a wide road"], "expected": "wide"}, {"args": ["one"], "expected": "one"}, {"args": [""], "expected": ""}],
        "subsequence": [{"args": ["ace", "abcde"], "expected": True}, {"args": ["aec", "abcde"], "expected": False}, {"args": ["", "x"], "expected": True}],
        "word_count": [{"args": ["one two one"], "expected": 3}, {"args": [""], "expected": 0}, {"args": ["  spaced  "], "expected": 1}],
        "acronym": [{"args": ["central processing unit"], "expected": "CPU"}, {"args": ["one"], "expected": "O"}, {"args": [""], "expected": ""}],
        "strip_vowels": [{"args": ["python"], "expected": "pythn"}, {"args": ["AEIOU"], "expected": ""}, {"args": ["sky"], "expected": "sky"}],
        "max_list": [{"args": [[1, 5, 3]], "expected": 5}, {"args": [[-4, -1, -9]], "expected": -1}, {"args": [[7]], "expected": 7}],
        "sum_list": [{"args": [[1, 2, 3]], "expected": 6}, {"args": [[]], "expected": 0}, {"args": [[-2, 5]], "expected": 3}],
        "unique_list": [{"args": [[2, 1, 2, 3, 1]], "expected": [2, 1, 3]}, {"args": [[],], "expected": []}, {"args": [["a", "a"]], "expected": ["a"]}],
        "chunk": [{"args": [[1, 2, 3, 4, 5], 2], "expected": [[1, 2], [3, 4], [5]]}, {"args": [[], 3], "expected": []}, {"args": [[1], 1], "expected": [[1]]}],
        "rotate_list": [{"args": [[1, 2, 3], 1], "expected": [3, 1, 2]}, {"args": [[1, 2], 4], "expected": [1, 2]}, {"args": [[], 3], "expected": []}],
        "flatten": [{"args": [[[1, 2], [3], []]], "expected": [1, 2, 3]}, {"args": [[],], "expected": []}, {"args": [[[], [4, 5]],], "expected": [4, 5]}],
        "running": [{"args": [[1, 2, 3]], "expected": [1, 3, 6]}, {"args": [[],], "expected": []}, {"args": [[-1, 1]], "expected": [-1, 0]}],
        "interleave": [{"args": [[1, 3], [2, 4, 6]], "expected": [1, 2, 3, 4, 6]}, {"args": [[], [1]], "expected": [1]}, {"args": [[1], []], "expected": [1]}],
        "two_sum": [{"args": [[2, 7, 11, 15], 9], "expected": True}, {"args": [[1, 2], 8], "expected": False}, {"args": [[3, 3], 6], "expected": True}],
        "transpose": [{"args": [[[1, 2], [3, 4], [5, 6]]], "expected": [[1, 3, 5], [2, 4, 6]]}, {"args": [[],], "expected": []}, {"args": [[[7]],], "expected": [[7]]}],
        "window": [{"args": [[1, 5, 2, 4], 2], "expected": [5, 5, 4]}, {"args": [[3], 1], "expected": [3]}, {"args": [[1, 2], 3], "expected": []}],
        "partition": [{"args": [[1, 2, 3, 4], 2], "expected": [[1, 2], [3, 4]]}, {"args": [[1, 2, 3], 1], "expected": [[1], [2, 3]]}, {"args": [[], 4], "expected": [[], []]}],
        "histogram": [{"args": ["Tea tea"], "expected": {"t": 2, "e": 2, "a": 2}}, {"args": [""], "expected": {}}, {"args": ["A!a"], "expected": {"a": 2}}],
        "invert": [{"args": [{"a": 1, "b": 2}], "expected": {"1": "a", "2": "b"}}, {"args": [{},], "expected": {}}, {"args": [{"x": 9}], "expected": {"9": "x"}}],
        "merge_counts": [{"args": [{"a": 1}, {"a": 2, "b": 3}], "expected": {"a": 3, "b": 3}}, {"args": [{}, {"x": 1}], "expected": {"x": 1}}, {"args": [{"z": 4}, {"z": -1}], "expected": {"z": 3}}],
        "common_keys": [{"args": [{"a": 1, "b": 2}, {"b": 5, "c": 6}], "expected": ["b"]}, {"args": [{}, {"a": 1}], "expected": []}, {"args": [{"x": 1}, {"x": 2}], "expected": ["x"]}],
        "group_parity": [{"args": [[1, 2, 3, 4]], "expected": {"even": [2, 4], "odd": [1, 3]}}, {"args": [[],], "expected": {"even": [], "odd": []}}, {"args": [[0]], "expected": {"even": [0], "odd": []}}],
        "lookup": [{"args": [{"a": 1}, "a", 0], "expected": 1}, {"args": [{}, "x", 7], "expected": 7}, {"args": [{"x": None}, "x", 2], "expected": None}],
        "sorted_items": [{"args": [{"b": 2, "a": 1}], "expected": [["a", 1], ["b", 2]]}, {"args": [{},], "expected": []}, {"args": [{"z": 0, "a": 3}], "expected": [["a", 3], ["z", 0]]}],
        "square_mapping": [{"args": [{"a": 2, "b": -3}], "expected": {"a": 4, "b": 9}}, {"args": [{},], "expected": {}}, {"args": [{"x": 0}], "expected": {"x": 0}}],
        "safe_int": [{"args": ["12", 0], "expected": 12}, {"args": ["bad", 7], "expected": 7}, {"args": ["-3", 1], "expected": -3}],
        "safe_div": [{"args": [6, 2, 0], "expected": 3.0}, {"args": [5, 0, -1], "expected": -1}, {"args": [0, 4, 9], "expected": 0.0}],
        "parse_bool": [{"args": ["true"], "expected": True}, {"args": ["NO"], "expected": False}, {"args": ["maybe"], "expected": None}],
        "mapping_default": [{"args": [{"a": 1}, "a", 0], "expected": 1}, {"args": [{}, "b", 0], "expected": 0}, {"args": [{"x": 2}, "b", 9], "expected": 9}],
        "validate_age": [{"args": [20], "expected": True}, {"args": [-1], "expected": False}, {"args": [130], "expected": False}],
        "binary_search": [{"args": [[1, 3, 5, 7], 5], "expected": 2}, {"args": [[1, 3], 2], "expected": -1}, {"args": [[], 1], "expected": -1}],
        "insertion_sort": [{"args": [[3, 1, 2]], "expected": [1, 2, 3]}, {"args": [[],], "expected": []}, {"args": [[2, 2, 1]], "expected": [1, 2, 2]}],
        "merge_sorted": [{"args": [[1, 4], [2, 3]], "expected": [1, 2, 3, 4]}, {"args": [[], [1]], "expected": [1]}, {"args": [[0], []], "expected": [0]}],
        "merge_intervals": [{"args": [[[1, 3], [2, 5], [8, 9]]], "expected": [[1, 5], [8, 9]]}, {"args": [[],], "expected": []}, {"args": [[[4, 6], [1, 2]]], "expected": [[1, 2], [4, 6]]}],
        "two_sum_indices": [{"args": [[2, 7, 11, 15], 9], "expected": [0, 1]}, {"args": [[3, 2, 4], 6], "expected": [1, 2]}, {"args": [[1, 2], 8], "expected": []}],
        "hamming": [{"args": ["karolin", "kathrin"], "expected": 3}, {"args": ["", ""], "expected": 0}, {"args": ["a", "b"], "expected": 1}],
        "edit": [{"args": ["kitten", "sitting"], "expected": 3}, {"args": ["", "abc"], "expected": 3}, {"args": ["same", "same"], "expected": 0}],
        "reachable": [{"args": [{"a": ["b"], "b": ["c"], "c": []}, "a", "c"], "expected": True}, {"args": [{"a": []}, "a", "z"], "expected": False}, {"args": [{}, "x", "x"], "expected": True}],
        "inversions": [{"args": [[2, 4, 1, 3, 5]], "expected": 3}, {"args": [[],], "expected": 0}, {"args": [[1, 2]], "expected": 0}],
        "kth": [{"args": [[4, 1, 3, 2], 2], "expected": 2}, {"args": [[9], 1], "expected": 9}, {"args": [[3, 1, 2], 3], "expected": 3}],
        "balanced": [{"args": ["([]{})"], "expected": True}, {"args": ["([)]"], "expected": False}, {"args": [""], "expected": True}],
        "longest_run": [{"args": [[3, 3, 1, 1, 1, 2]], "expected": 3}, {"args": [[],], "expected": 0}, {"args": [[5]], "expected": 1}],
        "diagonal": [{"args": [[[1, 2], [3, 4]]], "expected": 5}, {"args": [[[7]],], "expected": 7}, {"args": [[],], "expected": 0}],
        "change": [{"args": [[1, 2, 5], 5], "expected": 4}, {"args": [[2], 3], "expected": 0}, {"args": [[], 0], "expected": 1}],
        "point": [{"args": [[0, 0], [3, 4]], "expected": 5.0}, {"args": [[1, 1], [1, 1]], "expected": 0.0}, {"args": [[-1, -1], [2, 3]], "expected": 5.0}],
        "counter": [{"args": [0, [1, 2, -1]], "expected": [1, 3, 2]}, {"args": [5, [],], "expected": []}, {"args": [-2, [2, 2]], "expected": [0, 2]}],
        "rectangle": [{"args": [3, 4], "expected": 12}, {"args": [0, 9], "expected": 0}, {"args": [5, 2], "expected": 10}],
        "stack": [{"args": [["a", "b", "c"]], "expected": ["c", "b", "a"]}, {"args": [[],], "expected": []}, {"args": [[1]], "expected": [1]}],
        "inventory": [{"args": [{"a": 2, "b": 3}], "expected": 5}, {"args": [{},], "expected": 0}, {"args": [{"x": -1, "y": 4}], "expected": 3}],
        "bank": [{"args": [10, [5, -3, 2]], "expected": [15, 12, 14]}, {"args": [0, [],], "expected": []}, {"args": [4, [-4]], "expected": [0]}],
        "squares": [{"args": [[1, 2, 3]], "expected": [1, 4, 9]}, {"args": [[],], "expected": []}, {"args": [[-2, 0]], "expected": [4, 0]}],
        "evens": [{"args": [[1, 2, 4, 5]], "expected": [2, 4]}, {"args": [[],], "expected": []}, {"args": [[0, 3]], "expected": [0]}],
        "absolutes": [{"args": [[-2, 0, 3]], "expected": [2, 0, 3]}, {"args": [[],], "expected": []}, {"args": [[-9]], "expected": [9]}],
        "lengths": [{"args": [["a", "two"]], "expected": [1, 3]}, {"args": [[],], "expected": []}, {"args": [["GenPy"]], "expected": [5]}],
        "pair_map": [{"args": [[["a", 1], ["b", 2]]], "expected": {"a": 1, "b": 2}}, {"args": [[],], "expected": {}}, {"args": [[['x', 0]],], "expected": {"x": 0}}],
        "roman": [{"args": ["IV"], "expected": 4}, {"args": ["IX"], "expected": 9}, {"args": ["XII"], "expected": 12}],
        "version": [{"args": ["1.10.2", "1.2.9"], "expected": 1}, {"args": ["1.0", "1.0.0"], "expected": 0}, {"args": ["2", "10"], "expected": -1}],
        "decode_runs": [{"args": ["a3b2"], "expected": "aaabb"}, {"args": ["x1"], "expected": "x"}, {"args": [""], "expected": ""}],
        "base_digits": [{"args": [10, 2], "expected": "1010"}, {"args": [0, 8], "expected": "0"}, {"args": [31, 16], "expected": "1f"}],
        "border": [{"args": [[[1, 2], [3, 4]]], "expected": [1, 2, 4, 3]}, {"args": [[[7]],], "expected": [7]}, {"args": [[],], "expected": []}],
        "nearest": [{"args": [[1, 5, 9], 6], "expected": 5}, {"args": [[2, 8], 7], "expected": 8}, {"args": [[4], 100], "expected": 4}],
        "snake": [{"args": ["Hello, World"], "expected": "hello_world"}, {"args": ["one two"], "expected": "one_two"}, {"args": ["Already_snake"], "expected": "already_snake"}],
        "unique_window": [{"args": ["abcabcbb"], "expected": 3}, {"args": ["bbbbb"], "expected": 1}, {"args": [""], "expected": 0}],
        "prefix_products": [{"args": [[1, 2, 3, 4]], "expected": [1, 2, 6, 24]}, {"args": [[],], "expected": []}, {"args": [[5]], "expected": [5]}],
        "median": [{"args": [[1, 3, 2]], "expected": 2}, {"args": [[1, 2, 3, 4]], "expected": 2.5}, {"args": [[-2]], "expected": -2}],
    }
    if kind not in table:
        raise KeyError(f"No functional cases for {kind}")
    return table[kind]


def code_for(kind: str, function: str) -> str:
    simple = {
        "parity": f"def {function}(n):\n    return n % 2 == 0",
        "sign": f"def {function}(n):\n    return 'positive' if n > 0 else 'negative' if n < 0 else 'zero'",
        "clamp": f"def {function}(value, low, high):\n    return max(low, min(high, value))",
        "leap": f"def {function}(year):\n    return year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)",
        "divisible": f"def {function}(n, divisor):\n    return divisor != 0 and n % divisor == 0 if divisor else False",
        "band": f"def {function}(score):\n    return 'A' if score >= 90 else 'B' if score >= 80 else 'C' if score >= 70 else 'D' if score >= 60 else 'F'",
        "triangle": f"def {function}(a, b, c):\n    return a + b > c and a + c > b and b + c > a",
        "xor": f"def {function}(left, right):\n    return bool(left) != bool(right)",
        "factorial": f"def {function}(n):\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result",
        "digit_sum": f"def {function}(n):\n    return sum(int(value) for value in str(abs(n)))",
        "gcd": f"def {function}(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)",
        "lcm": f"def {function}(a, b):\n    if not a or not b:\n        return 0\n    x, y = abs(a), abs(b)\n    while y:\n        x, y = y, x % y\n    return abs(a * b) // x",
        "prime": f"def {function}(n):\n    if n < 2:\n        return False\n    divisor = 2\n    while divisor * divisor <= n:\n        if n % divisor == 0:\n            return False\n        divisor += 1\n    return True",
        "fib": f"def {function}(n):\n    first, second = 0, 1\n    for _ in range(n):\n        first, second = second, first + second\n    return first",
        "sum_multiples": f"def {function}(limit, divisor):\n    return sum(range(divisor, limit + 1, divisor)) if divisor > 0 else 0",
        "digits": f"def {function}(n):\n    return len(str(abs(n)))",
        "collatz": f"def {function}(n):\n    steps = 0\n    while n > 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        steps += 1\n    return steps",
        "power": f"def {function}(base, exponent):\n    result = 1\n    for _ in range(exponent):\n        result *= base\n    return result",
        "reverse": f"def {function}(text):\n    return text[::-1]",
        "palindrome": f"def {function}(text):\n    return text == text[::-1]",
        "vowels": f"def {function}(text):\n    return sum(character.lower() in 'aeiou' for character in text)",
        "first_unique": f"def {function}(text):\n    for character in text:\n        if text.count(character) == 1:\n            return character\n    return None",
        "anagram": f"def {function}(left, right):\n    return sorted(left) == sorted(right)",
        "rotate_text": f"def {function}(text, amount):\n    return text[-amount % len(text):] + text[:-amount % len(text)] if text else text",
        "compress": f"def {function}(text):\n    result = []\n    index = 0\n    while index < len(text):\n        end = index + 1\n        while end < len(text) and text[end] == text[index]:\n            end += 1\n        result.append(text[index] + str(end - index))\n        index = end\n    return ''.join(result)",
        "longest_word": f"def {function}(text):\n    words = text.split()\n    return max(words, key=len) if words else ''",
        "subsequence": f"def {function}(small, large):\n    iterator = iter(large)\n    return all(character in iterator for character in small)",
        "word_count": f"def {function}(text):\n    return len(text.split())",
        "acronym": f"def {function}(text):\n    return ''.join(word[0].upper() for word in text.split())",
        "strip_vowels": f"def {function}(text):\n    return ''.join(character for character in text if character.lower() not in 'aeiou')",
        "max_list": f"def {function}(values):\n    return max(values)",
        "sum_list": f"def {function}(values):\n    return sum(values)",
        "unique_list": f"def {function}(values):\n    result = []\n    for value in values:\n        if value not in result:\n            result.append(value)\n    return result",
        "chunk": f"def {function}(values, size):\n    return [values[index:index + size] for index in range(0, len(values), size)]",
        "rotate_list": f"def {function}(values, amount):\n    if not values:\n        return []\n    amount %= len(values)\n    return values[-amount:] + values[:-amount] if amount else list(values)",
        "flatten": f"def {function}(values):\n    return [item for group in values for item in group]",
        "running": f"def {function}(values):\n    result, total = [], 0\n    for value in values:\n        total += value\n        result.append(total)\n    return result",
        "interleave": f"def {function}(left, right):\n    result = []\n    for index in range(max(len(left), len(right))):\n        if index < len(left): result.append(left[index])\n        if index < len(right): result.append(right[index])\n    return result",
        "two_sum": f"def {function}(values, target):\n    return any(values[left] + values[right] == target for left in range(len(values)) for right in range(left + 1, len(values)))",
        "transpose": f"def {function}(rows):\n    return [list(column) for column in zip(*rows)] if rows else []",
        "window": f"def {function}(values, width):\n    return [max(values[index:index + width]) for index in range(len(values) - width + 1)] if width > 0 else []",
        "partition": f"def {function}(values, pivot):\n    return [values[:pivot], values[pivot:]]",
        "histogram": f"def {function}(text):\n    result = {{}}\n    for character in text.lower():\n        if character.isalpha(): result[character] = result.get(character, 0) + 1\n    return result",
        "invert": f"def {function}(mapping):\n    return {{str(value): key for key, value in mapping.items()}}",
        "merge_counts": f"def {function}(left, right):\n    result = dict(left)\n    for key, value in right.items(): result[key] = result.get(key, 0) + value\n    return result",
        "common_keys": f"def {function}(left, right):\n    return sorted(set(left) & set(right))",
        "group_parity": f"def {function}(values):\n    return {{'even': [value for value in values if value % 2 == 0], 'odd': [value for value in values if value % 2]}}",
        "lookup": f"def {function}(mapping, key, default):\n    return mapping.get(key, default)",
        "sorted_items": f"def {function}(mapping):\n    return [[key, mapping[key]] for key in sorted(mapping)]",
        "square_mapping": f"def {function}(mapping):\n    return {{key: value * value for key, value in mapping.items()}}",
        "safe_int": f"def {function}(text, default):\n    try: return int(text)\n    except (TypeError, ValueError): return default",
        "safe_div": f"def {function}(numerator, denominator, default):\n    try: return numerator / denominator\n    except ZeroDivisionError: return default",
        "parse_bool": f"def {function}(text):\n    value = text.strip().lower()\n    return True if value in ('true', 'yes', '1') else False if value in ('false', 'no', '0') else None",
        "mapping_default": f"def {function}(mapping, key, default):\n    return mapping[key] if key in mapping else default",
        "validate_age": f"def {function}(age):\n    return 0 <= age <= 120",
        "binary_search": f"def {function}(values, target):\n    left, right = 0, len(values) - 1\n    while left <= right:\n        middle = (left + right) // 2\n        if values[middle] == target: return middle\n        if values[middle] < target: left = middle + 1\n        else: right = middle - 1\n    return -1",
        "insertion_sort": f"def {function}(values):\n    result = list(values)\n    for index in range(1, len(result)):\n        value = result[index]; cursor = index - 1\n        while cursor >= 0 and result[cursor] > value:\n            result[cursor + 1] = result[cursor]; cursor -= 1\n        result[cursor + 1] = value\n    return result",
        "merge_sorted": f"def {function}(left, right):\n    return sorted(list(left) + list(right))",
        "merge_intervals": f"def {function}(intervals):\n    result = []\n    for start, end in sorted(intervals):\n        if result and start <= result[-1][1]: result[-1][1] = max(result[-1][1], end)\n        else: result.append([start, end])\n    return result",
        "two_sum_indices": f"def {function}(values, target):\n    seen = {{}}\n    for index, value in enumerate(values):\n        if target - value in seen: return [seen[target - value], index]\n        seen[value] = index\n    return []",
        "hamming": f"def {function}(left, right):\n    return sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))",
        "edit": f"def {function}(left, right):\n    row = list(range(len(right) + 1))\n    for i, a in enumerate(left, 1):\n        next_row = [i]\n        for j, b in enumerate(right, 1): next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))\n        row = next_row\n    return row[-1]",
        "reachable": f"def {function}(graph, start, target):\n    todo, seen = [start], set()\n    while todo:\n        node = todo.pop(0)\n        if node == target: return True\n        if node not in seen: seen.add(node); todo.extend(graph.get(node, []))\n    return False",
        "inversions": f"def {function}(values):\n    return sum(values[i] > values[j] for i in range(len(values)) for j in range(i + 1, len(values)))",
        "kth": f"def {function}(values, k):\n    return sorted(values)[k - 1]",
        "balanced": "def " + function + "(text):\n    pairs = {')': '(', ']': '[', '}': '{'}; stack = []\n    for character in text:\n        if character in '([{': stack.append(character)\n        elif character in pairs and (not stack or stack.pop() != pairs[character]): return False\n    return not stack",
        "longest_run": f"def {function}(values):\n    best = current = 0; previous = object()\n    for value in values:\n        current = current + 1 if value == previous else 1; best = max(best, current); previous = value\n    return best",
        "diagonal": f"def {function}(matrix):\n    return sum(matrix[index][index] for index in range(len(matrix)))",
        "change": f"def {function}(coins, amount):\n    ways = [0] * (amount + 1); ways[0] = 1\n    for coin in coins:\n        for value in range(coin, amount + 1): ways[value] += ways[value - coin]\n    return ways[amount]",
        "point": f"def {function}(left, right):\n    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5",
        "counter": f"def {function}(start, increments):\n    value = start; result = []\n    for amount in increments: value += amount; result.append(value)\n    return result",
        "rectangle": f"def {function}(width, height):\n    return width * height",
        "stack": f"def {function}(values):\n    stack = list(values); result = []\n    while stack: result.append(stack.pop())\n    return result",
        "inventory": f"def {function}(inventory):\n    return sum(inventory.values())",
        "bank": f"def {function}(balance, changes):\n    result = []\n    for change in changes: balance += change; result.append(balance)\n    return result",
        "squares": f"def {function}(values):\n    return [value * value for value in values]",
        "evens": f"def {function}(values):\n    return [value for value in values if value % 2 == 0]",
        "absolutes": f"def {function}(values):\n    return [abs(value) for value in values]",
        "lengths": f"def {function}(values):\n    return [len(value) for value in values]",
        "pair_map": f"def {function}(pairs):\n    return {{key: value for key, value in pairs}}",
        "roman": f"def {function}(text):\n    values = {{'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}}; total = 0; previous = 0\n    for character in reversed(text):\n        value = values[character]; total += -value if value < previous else value; previous = max(previous, value)\n    return total",
        "version": f"def {function}(left, right):\n    a = [int(x) for x in left.split('.')]; b = [int(x) for x in right.split('.')]; size = max(len(a), len(b)); a += [0] * (size - len(a)); b += [0] * (size - len(b))\n    return (a > b) - (a < b)",
        "decode_runs": f"def {function}(text):\n    result = ''; index = 0\n    while index < len(text):\n        character = text[index]; index += 1; start = index\n        while index < len(text) and text[index].isdigit(): index += 1\n        result += character * int(text[start:index])\n    return result",
        "base_digits": f"def {function}(number, base):\n    digits = '0123456789abcdef'\n    if number == 0: return '0'\n    result = ''\n    while number: result = digits[number % base] + result; number //= base\n    return result",
        "border": f"def {function}(matrix):\n    if not matrix: return []\n    if len(matrix) == 1: return list(matrix[0])\n    top = list(matrix[0]); right = [row[-1] for row in matrix[1:-1]]; bottom = list(reversed(matrix[-1]))\n    left = [row[0] for row in reversed(matrix[1:-1])]\n    return top + right + bottom + left",
        "nearest": f"def {function}(values, target):\n    return min(values, key=lambda value: (abs(value - target), value))",
        "snake": f"def {function}(text):\n    return '_'.join(text.replace('-', ' ').replace('_', ' ').replace(',', ' ').lower().split())",
        "unique_window": f"def {function}(text):\n    best = 0\n    for start in range(len(text)):\n        seen = set()\n        for character in text[start:]:\n            if character in seen: break\n            seen.add(character); best = max(best, len(seen))\n    return best",
        "prefix_products": f"def {function}(values):\n    result = []; product = 1\n    for value in values: product *= value; result.append(product)\n    return result",
        "median": f"def {function}(values):\n    ordered = sorted(values); middle = len(ordered) // 2\n    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2",
    }
    return simple[kind]


KIND_BY_CATEGORY = {"conditionals": ["parity", "sign", "clamp", "leap", "divisible", "band", "triangle", "xor"], "loops_math": ["factorial", "digit_sum", "gcd", "lcm", "prime", "fib", "sum_multiples", "digits", "collatz", "power"], "strings": ["reverse", "palindrome", "vowels", "first_unique", "anagram", "rotate_text", "compress", "longest_word", "subsequence", "word_count", "acronym", "strip_vowels"], "sequences": ["max_list", "sum_list", "unique_list", "chunk", "rotate_list", "flatten", "running", "interleave", "two_sum", "transpose", "window", "partition"], "dictionaries_sets": ["histogram", "invert", "merge_counts", "common_keys", "group_parity", "lookup", "sorted_items", "square_mapping"], "functions_recursion": ["factorial", "sum_list", "fib", "flatten", "counter", "counter", "fib", "sum_list", "longest_run"], "comprehensions": ["squares", "evens", "absolutes", "lengths", "pair_map"], "oop": ["point", "counter", "rectangle", "stack", "inventory", "bank"], "exceptions": ["safe_int", "safe_div", "parse_bool", "mapping_default", "validate_age"], "algorithms": ["binary_search", "insertion_sort", "merge_sorted", "merge_intervals", "two_sum_indices", "hamming", "edit", "reachable", "inversions", "kth", "balanced", "longest_run", "diagonal", "change"], "debugging": ["max_list", "reverse", "parity", "unique_list", "binary_search", "factorial", "chunk"]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocate(skills: list[str], total: int) -> list[tuple[str, int]]:
    base, remainder = divmod(total, len(skills))
    return [(skill, base + (index < remainder)) for index, skill in enumerate(skills)]


def make_record(split: str, category: str, skill: str, kind: str, index: int, style_index: int, random_tag: int, challenge: bool = False, seed: int = 42) -> dict:
    function = fname(skill, split, index)
    style = STYLES[style_index]
    description = skill.replace("_", " ")
    prefixes = [f"Implement {function} for {description}; use the stated Python behavior and return a deterministic result.", f"Create {function} that handles {description} cases with ordinary Python values.", f"Given a small Python task about {description}, complete the function {function}.", f"Write {function} under this constraint: solve the {description} problem without external resources."]
    instruction = prefixes[style_index] + f" Scenario {random_tag}."
    input_text = ""
    if category == "debugging":
        input_text = f"def broken_{function}(values):\n    return values\n"
        instruction = f"Fix the following Python function so it correctly handles {description}; return the complete corrected function named {function}. Scenario {random_tag}."
    return {"id": f"v2_{split}_{index:06d}", "instruction": instruction, "input": input_text, "response": code_for(kind, function), "response_type": "code", "category": category, "skill_id": f"{category}.{skill}", "prompt_template_id": f"{split}.{category}.{skill}.{style}.v2", "difficulty": ["easy", "intermediate", "hard"][index % 3], "task_style": "debugging" if category == "debugging" else "code_generation", "function_name": function, "test_cases": cases(kind), "provenance": {"generator": GENERATOR_VERSION, "seed": seed, "random_tag": random_tag}}


def sanity_records() -> list[dict]:
    prompts = [
        ("Write Python code to check whether a number is even or odd.", "def solve(n):\n    return 'Even' if n % 2 == 0 else 'Odd'", [{"args": [2], "expected": "Even"}, {"args": [7], "expected": "Odd"}, {"args": [0], "expected": "Even"}, {"args": [-4], "expected": "Even"}, {"args": [-3], "expected": "Odd"}]),
        ("Write a Python function to reverse a string.", "def solve(text):\n    return text[::-1]", [{"args": ["hello"], "expected": "olleh"}, {"args": [""], "expected": ""}, {"args": ["a"], "expected": "a"}, {"args": ["GenPy"], "expected": "yPneG"}]),
        ("Write Python code to find the largest number in a list.", "def solve(values):\n    return max(values)", [{"args": [[1, 5, 3]], "expected": 5}, {"args": [[-4, -1, -9]], "expected": -1}, {"args": [[7]], "expected": 7}, {"args": [[2, 2, 2]], "expected": 2}]),
    ]
    extra = [("Check whether a string is a palindrome.", "palindrome"), ("Calculate the factorial of an integer.", "factorial"), ("Remove repeated values while preserving order.", "unique_list"), ("Count vowels in a string.", "vowels"), ("Sort a list without calling sort.", "insertion_sort"), ("Calculate a Fibonacci number.", "fib"), ("Find the greatest common divisor.", "gcd"), ("Check whether a number is prime.", "prime"), ("Find the first unique character.", "first_unique"), ("Merge two sorted lists.", "merge_sorted"), ("Compute a running sum.", "running"), ("Check balanced brackets.", "balanced"), ("Find an item with binary search.", "binary_search"), ("Count words in text.", "word_count"), ("Compute edit distance.", "edit"), ("Rotate a list rightward.", "rotate_list"), ("Convert a number to base digits.", "base_digits")]
    records = []
    for index, (prompt, response, tests) in enumerate(prompts + [(item[0], code_for(item[1], "solve"), cases(item[1])) for item in extra], 1):
        records.append({"id": f"v2_sanity_{index:03d}", "instruction": prompt, "input": "", "response": response, "response_type": "code", "category": "sanity", "skill_id": f"sanity.{index}", "prompt_template_id": f"sanity.frozen.{index:03d}", "difficulty": "easy", "task_style": "evaluation_only", "function_name": "solve", "test_cases": tests, "provenance": {"generator": GENERATOR_VERSION, "seed": 42, "evaluation_only": True}})
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_split(split: str, total: int, seed: int, challenge_only: bool = False) -> list[dict]:
    rng = random.Random(seed + {"train": 0, "validation": 1001, "challenge": 2002}[split])
    rows = []
    if challenge_only:
        skills = [("challenge", skill, skill) for skill in CHALLENGE_ONLY]
        for index, (_, skill, kind) in enumerate((item for skill, kind in zip(CHALLENGE_ONLY, CHALLENGE_ONLY_KINDS) for item in [("challenge", skill, kind)]), 1):
            count = total // len(CHALLENGE_ONLY) + (index <= total % len(CHALLENGE_ONLY))
            for local in range(count):
                rows.append(make_record(split, "challenge", skill, kind, len(rows) + 1, local % 4, rng.randrange(100000, 999999), True, seed))
        return rows
    category_remaining = {category: round(total * target / 10000) for category, target in CATEGORY_TARGETS.items()}
    category_remaining[next(reversed(category_remaining))] += total - sum(category_remaining.values())
    for category, category_total in category_remaining.items():
        skills = CATEGORY_SKILLS[category]
        kind_pool = KIND_BY_CATEGORY[category]
        for skill_index, (skill, count) in enumerate(allocate(skills, category_total)):
            kind = kind_pool[skill_index % len(kind_pool)]
            for local in range(count):
                rows.append(make_record(split, category, skill, kind, len(rows) + 1, local % 4, rng.randrange(100000, 999999), seed=seed))
    rng.shuffle(rows)
    return rows[:total]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/instruction/python_v2")
    parser.add_argument("--train-count", type=int, default=10000)
    parser.add_argument("--validation-count", type=int, default=1000)
    parser.add_argument("--challenge-count", type=int, default=500)
    args = parser.parse_args()
    if (args.train_count, args.validation_count, args.challenge_count) != (10000, 1000, 500):
        raise ValueError("Checkpoint 8-v2 pilot counts are fixed at 10000/1000/500")
    output = Path(args.output_dir)
    train = build_split("train", args.train_count, args.seed)
    validation = build_split("validation", args.validation_count, args.seed)
    challenge_normal = build_split("challenge", 350, args.seed)
    challenge_new = build_split("challenge", 150, args.seed + 77, True)
    challenge = challenge_normal + challenge_new
    for index, row in enumerate(challenge, 1):
        row["id"] = f"v2_challenge_{index:06d}"
    random.Random(args.seed + 99).shuffle(challenge)
    sanity = sanity_records()
    for name, rows in (("train", train), ("validation", validation), ("challenge", challenge), ("sanity", sanity)):
        write_jsonl(output / f"{name}.jsonl", rows)
    manifest = {"dataset_name": "GenPy-SFT-v2-Pilot", "dataset_version": "genpy-sft-v2-pilot-v1", "generator_version": GENERATOR_VERSION, "seed": args.seed, "train_count": len(train), "validation_count": len(validation), "challenge_count": len(challenge), "sanity_count": len(sanity), "cp7_known_source_hashes": {"train": CP7_TRAIN_HASH, "validation": CP7_VALIDATION_HASH}, "fresh_deterministic_synthetic": True, "external_dataset_used": False, "internet_used": False, "sft_training_sources": ["train.jsonl", "validation.jsonl"], "challenge_policy": "frozen evaluation only; never SFT training or hyperparameter selection", "sanity_policy": "frozen evaluation only; never SFT training or validation", "files": {name: {"sha256": sha256(output / f"{name}.jsonl"), "count": len(rows)} for name, rows in (("train", train), ("validation", validation), ("challenge", challenge), ("sanity", sanity))}, "category_distribution": dict(Counter(row["category"] for row in train)), "skill_count": len({row["skill_id"] for row in train}), "template_count": len({row["prompt_template_id"] for row in train}), "provenance": "Independently constructed semantic task catalog; CP7 files are only used by later contamination audits."}
    (output / "DATASET_V2_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(train), "validation": len(validation), "challenge": len(challenge), "sanity": len(sanity), "skills": manifest["skill_count"], "templates": manifest["template_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
