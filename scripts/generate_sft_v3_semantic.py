"""Generate the deterministic, semantic GenPy Checkpoint 8-v3 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

VERSION = "genpy-sft-v3-semantic-v1"
GENERATOR = "genpy-sft-v3-semantic-v1"
SEED = 42
REQUIRED = {"id", "instruction", "input", "response", "response_type", "category", "skill_id", "prompt_template_id", "difficulty", "task_style", "function_name", "test_cases", "provenance"}


def tc(args, expected):
    return {"args": args, "kwargs": {}, "expected": expected}


def item(category, skill, description, code, cases):
    return {"category": category, "skill": skill, "description": description, "code": code, "cases": cases}


CATALOG = [
    item("conditionals", "parity_even", "takes an integer n and returns True when n is even and False otherwise", "def solve(n):\n    return n % 2 == 0", [tc([2], True), tc([7], False), tc([0], True), tc([-4], True), tc([-3], False)]),
    item("conditionals", "sign_label", "takes an integer n and returns 'positive', 'negative', or 'zero' according to its sign", "def solve(n):\n    return 'positive' if n > 0 else 'negative' if n < 0 else 'zero'", [tc([-5], "negative"), tc([0], "zero"), tc([8], "positive"), tc([-1], "negative"), tc([100], "positive")]),
    item("conditionals", "clamp_interval", "takes value, low, and high and returns value limited to the inclusive interval from low to high", "def solve(value, low, high):\n    return max(low, min(high, value))", [tc([4, 0, 10], 4), tc([-2, 0, 10], 0), tc([12, 0, 10], 10), tc([0, 0, 10], 0), tc([10, 0, 10], 10)]),
    item("conditionals", "leap_year", "takes a year and returns True exactly for Gregorian leap years", "def solve(year):\n    return year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)", [tc([2000], True), tc([1900], False), tc([2024], True), tc([2023], False), tc([2400], True)]),
    item("conditionals", "divisible_by", "takes integers n and divisor and returns whether n is divisible by a nonzero divisor; zero divisor returns False", "def solve(n, divisor):\n    return divisor != 0 and n % divisor == 0", [tc([12, 3], True), tc([14, 3], False), tc([0, 5], True), tc([9, 0], False), tc([-12, 4], True)]),
    item("basic_math", "factorial", "takes a nonnegative integer n and returns n factorial, with factorial of zero equal to one", "def solve(n):\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result", [tc([0], 1), tc([1], 1), tc([5], 120), tc([7], 5040), tc([3], 6)]),
    item("basic_math", "digit_sum", "takes an integer n and returns the sum of the decimal digits of its absolute value", "def solve(n):\n    return sum(int(digit) for digit in str(abs(n)))", [tc([0], 0), tc([1234], 10), tc([-58], 13), tc([90001], 10), tc([7], 7)]),
    item("basic_math", "digit_count", "takes an integer n and returns the number of decimal digits in its absolute value, counting zero as one digit", "def solve(n):\n    return len(str(abs(n)))", [tc([0], 1), tc([42], 2), tc([-987], 3), tc([10000], 5), tc([6], 1)]),
    item("basic_math", "gcd", "takes integers a and b and returns their nonnegative greatest common divisor", "def solve(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)", [tc([18, 24], 6), tc([7, 3], 1), tc([0, 9], 9), tc([-12, 8], 4), tc([21, 0], 21)]),
    item("basic_math", "lcm", "takes integers a and b and returns their nonnegative least common multiple, returning zero when either input is zero", "def solve(a, b):\n    if a == 0 or b == 0:\n        return 0\n    x, y = abs(a), abs(b)\n    while y:\n        x, y = y, x % y\n    return abs(a * b) // x", [tc([4, 6], 12), tc([5, 7], 35), tc([0, 4], 0), tc([-3, 8], 24), tc([12, 18], 36)]),
    item("basic_math", "prime_check", "takes an integer n and returns True when n is prime; values below two are not prime", "def solve(n):\n    if n < 2:\n        return False\n    divisor = 2\n    while divisor * divisor <= n:\n        if n % divisor == 0:\n            return False\n        divisor += 1\n    return True", [tc([-1], False), tc([0], False), tc([2], True), tc([17], True), tc([49], False)]),
    item("basic_math", "fibonacci", "takes a nonnegative index n and returns the nth Fibonacci number with F(0)=0 and F(1)=1", "def solve(n):\n    first, second = 0, 1\n    for _ in range(n):\n        first, second = second, first + second\n    return first", [tc([0], 0), tc([1], 1), tc([2], 1), tc([8], 21), tc([10], 55)]),
    item("basic_math", "integer_power", "takes base and nonnegative exponent and returns base raised to that integer exponent", "def solve(base, exponent):\n    result = 1\n    for _ in range(exponent):\n        result *= base\n    return result", [tc([2, 0], 1), tc([3, 2], 9), tc([5, 3], 125), tc([-2, 3], -8), tc([10, 1], 10)]),
    item("strings", "reverse_text", "takes a string text and returns its characters in reverse order, including the empty string case", "def solve(text):\n    return text[::-1]", [tc(["hello"], "olleh"), tc([""], ""), tc(["a"], "a"), tc(["GenPy"], "yPneG"), tc(["a b"], "b a")]),
    item("strings", "palindrome", "takes a string text and returns True when it reads identically forward and backward, with case and spaces significant", "def solve(text):\n    return text == text[::-1]", [tc(["level"], True), tc(["python"], False), tc([""], True), tc(["Aa"], False), tc(["a b a"], True)]),
    item("strings", "vowel_count", "takes a string text and counts the English vowels a, e, i, o, and u without changing case significance for other characters", "def solve(text):\n    return sum(character.lower() in 'aeiou' for character in text)", [tc(["Umbrella"], 3), tc(["rhythm"], 0), tc(["AEIOU"], 5), tc([""], 0), tc(["Python 3"], 1)]),
    item("strings", "anagram", "takes strings left and right and returns True when they contain the same characters with the same multiplicities", "def solve(left, right):\n    return sorted(left) == sorted(right)", [tc(["listen", "silent"], True), tc(["cat", "car"], False), tc(["", ""], True), tc(["aab", "aba"], True), tc(["ab", "a"], False)]),
    item("strings", "first_unique_character", "takes text and returns its first character occurring exactly once, or None when no such character exists", "def solve(text):\n    for character in text:\n        if text.count(character) == 1:\n            return character\n    return None", [tc(["swiss"], "w"), tc(["aabb"], None), tc(["z"], "z"), tc(["level"], "v"), tc([""], None)]),
    item("strings", "word_count", "takes text and returns the number of whitespace-separated words, ignoring leading and trailing whitespace", "def solve(text):\n    return len(text.split())", [tc(["one two one"], 3), tc([""], 0), tc(["  spaced  "], 1), tc(["a b c"], 3), tc(["one\ttwo"], 2)]),
    item("strings", "strip_vowels", "takes text and returns it with English vowels removed while preserving all other characters and order", "def solve(text):\n    return ''.join(character for character in text if character.lower() not in 'aeiou')", [tc(["python"], "pythn"), tc(["AEIOU"], ""), tc(["sky"], "sky"), tc(["hello world"], "hll wrld"), tc([""], "")]),
    item("strings", "compress_runs", "takes text and replaces each consecutive run by its character followed by the run length", "def solve(text):\n    result = []\n    index = 0\n    while index < len(text):\n        end = index + 1\n        while end < len(text) and text[end] == text[index]:\n            end += 1\n        result.append(text[index] + str(end - index))\n        index = end\n    return ''.join(result)", [tc(["aaabbc"], "a3b2c1"), tc(["x"], "x1"), tc([""], ""), tc(["1111"], "14"), tc(["ab"], "a1b1")]),
    item("lists_sequences", "maximum_item", "takes a nonempty list values and returns its largest item", "def solve(values):\n    return max(values)", [tc([[1, 5, 3]], 5), tc([[-4, -1, -9]], -1), tc([[7]], 7), tc([[0, 0]], 0), tc([[10, 2, 8]], 10)]),
    item("lists_sequences", "sum_items", "takes a list values and returns the sum of its items, returning zero for an empty list", "def solve(values):\n    return sum(values)", [tc([[1, 2, 3]], 6), tc([[]], 0), tc([[-2, 5]], 3), tc([[0]], 0), tc([[10, -3, 2]], 9)]),
    item("lists_sequences", "stable_unique", "takes values and removes later duplicates while preserving the first-seen order", "def solve(values):\n    result = []\n    for value in values:\n        if value not in result:\n            result.append(value)\n    return result", [tc([[2, 1, 2, 3, 1]], [2, 1, 3]), tc([[]], []), tc([["a", "a"]], ["a"]), tc([[3, 3, 2]], [3, 2]), tc([[1, 2, 1, 2]], [1, 2])]),
    item("lists_sequences", "running_totals", "takes values and returns a list whose each item is the cumulative sum through that position", "def solve(values):\n    result = []\n    total = 0\n    for value in values:\n        total += value\n        result.append(total)\n    return result", [tc([[1, 2, 3]], [1, 3, 6]), tc([[]], []), tc([[-1, 1]], [-1, 0]), tc([[5]], [5]), tc([[2, -2, 4]], [2, 0, 4])]),
    item("lists_sequences", "rotate_sequence", "takes a list values and integer amount and rotates the list to the right by amount positions; empty lists stay empty", "def solve(values, amount):\n    if not values:\n        return []\n    amount %= len(values)\n    return values[-amount:] + values[:-amount] if amount else list(values)", [tc([[1, 2, 3], 1], [3, 1, 2]), tc([[1, 2], 4], [1, 2]), tc([[], 3], []), tc([[1, 2, 3], 0], [1, 2, 3]), tc([[1, 2, 3], -1], [2, 3, 1])]),
    item("lists_sequences", "flatten_one_level", "takes a list of lists values and concatenates exactly one level of nested lists", "def solve(values):\n    return [item for group in values for item in group]", [tc([[[1, 2], [3], []]], [1, 2, 3]), tc([[]], []), tc([[[], [4, 5]]], [4, 5]), tc([[["a"], ["b", "c"]]], ["a", "b", "c"]), tc([[[1], []]], [1])]),
    item("lists_sequences", "chunk_sequence", "takes values and positive size and returns consecutive sublists of at most size items", "def solve(values, size):\n    return [values[index:index + size] for index in range(0, len(values), size)]", [tc([[1, 2, 3, 4, 5], 2], [[1, 2], [3, 4], [5]]), tc([[], 3], []), tc([[1], 1], [[1]]), tc([[1, 2, 3], 5], [[1, 2, 3]]), tc([[1, 2, 3, 4], 2], [[1, 2], [3, 4]])]),
    item("lists_sequences", "second_largest", "takes a list with at least two distinct values and returns its second-largest distinct value; otherwise returns None", "def solve(values):\n    unique = sorted(set(values))\n    return unique[-2] if len(unique) >= 2 else None", [tc([[1, 5, 3]], 3), tc([[-4, -1, -9]], -4), tc([[7]], None), tc([[2, 2]], None), tc([[10, 3, 10, 7]], 7)]),
    item("lists_sequences", "two_sum_exists", "takes values and target and returns True when two different positions sum to target", "def solve(values, target):\n    seen = set()\n    for value in values:\n        if target - value in seen:\n            return True\n        seen.add(value)\n    return False", [tc([[2, 7, 11, 15], 9], True), tc([[1, 2], 8], False), tc([[3, 3], 6], True), tc([[], 0], False), tc([[-2, 5, 7], 3], True)]),
    item("dictionaries_sets", "character_histogram", "takes text and returns a dictionary counting alphabetic characters case-insensitively, ignoring nonletters", "def solve(text):\n    result = {}\n    for character in text.lower():\n        if character.isalpha():\n            result[character] = result.get(character, 0) + 1\n    return result", [tc(["Tea tea"], {"t": 2, "e": 2, "a": 2}), tc([""], {}), tc(["A!a"], {"a": 2}), tc(["abc"], {"a": 1, "b": 1, "c": 1}), tc(["Hello"], {"h": 1, "e": 1, "l": 2, "o": 1})]),
    item("dictionaries_sets", "merge_counters", "takes dictionaries left and right and returns their sum by key, treating missing counts as zero", "def solve(left, right):\n    result = dict(left)\n    for key, value in right.items():\n        result[key] = result.get(key, 0) + value\n    return result", [tc([{"a": 1}, {"a": 2, "b": 3}], {"a": 3, "b": 3}), tc([{}, {"x": 1}], {"x": 1}), tc([{"z": 4}, {"z": -1}], {"z": 3}), tc([{"a": 0}, {}], {"a": 0}), tc([{"x": 2}, {"y": 5}], {"x": 2, "y": 5})]),
    item("dictionaries_sets", "common_keys", "takes dictionaries left and right and returns their shared keys in sorted order", "def solve(left, right):\n    return sorted(set(left) & set(right))", [tc([{"a": 1, "b": 2}, {"b": 5, "c": 6}], ["b"]), tc([{}, {"a": 1}], []), tc([{"x": 1}, {"x": 2}], ["x"]), tc([{"b": 1, "a": 2}, {"a": 0, "c": 1}], ["a"]), tc([{"z": 1}, {"y": 2}], [])]),
    item("dictionaries_sets", "invert_mapping", "takes a mapping with unique values and returns a mapping from each value converted to a string to its original key", "def solve(mapping):\n    return {str(value): key for key, value in mapping.items()}", [tc([{"a": 1, "b": 2}], {"1": "a", "2": "b"}), tc([{}], {}), tc([{"x": 9}], {"9": "x"}), tc([{"left": 0}], {"0": "left"}), tc([{"a": -2, "b": 5}], {"-2": "a", "5": "b"})]),
    item("basic_algorithms", "binary_search", "takes sorted values and target and returns the target index or -1 when absent", "def solve(values, target):\n    left, right = 0, len(values) - 1\n    while left <= right:\n        middle = (left + right) // 2\n        if values[middle] == target:\n            return middle\n        if values[middle] < target:\n            left = middle + 1\n        else:\n            right = middle - 1\n    return -1", [tc([[1, 3, 5, 7], 5], 2), tc([[1, 3], 2], -1), tc([[], 1], -1), tc([[0], 0], 0), tc([[2, 4, 6, 8, 10], 10], 4)]),
    item("basic_algorithms", "insertion_sort", "takes values and returns a new list sorted in ascending order without changing the input", "def solve(values):\n    result = list(values)\n    for index in range(1, len(result)):\n        value = result[index]\n        cursor = index - 1\n        while cursor >= 0 and result[cursor] > value:\n            result[cursor + 1] = result[cursor]\n            cursor -= 1\n        result[cursor + 1] = value\n    return result", [tc([[3, 1, 2]], [1, 2, 3]), tc([[]], []), tc([[2, 2, 1]], [1, 2, 2]), tc([[5, 4, 3, 2, 1]], [1, 2, 3, 4, 5]), tc([[-1, 0, -3]], [-3, -1, 0])]),
    item("basic_algorithms", "merge_sorted", "takes two already sorted lists and returns one sorted list containing all items", "def solve(left, right):\n    return sorted(list(left) + list(right))", [tc([[1, 4], [2, 3]], [1, 2, 3, 4]), tc([[], [1]], [1]), tc([[0], []], [0]), tc([[1, 1], [1]], [1, 1, 1]), tc([[-2, 4], [-1, 3]], [-2, -1, 3, 4])]),
    item("basic_algorithms", "balanced_brackets", "takes text containing brackets and returns True when (), [], and {} are correctly nested and matched", "def solve(text):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for character in text:\n        if character in '([{':\n            stack.append(character)\n        elif character in pairs:\n            if not stack or stack.pop() != pairs[character]:\n                return False\n    return not stack", [tc(["([]{})"], True), tc(["([)]"], False), tc([""], True), tc(["(("], False), tc(["{[()]}"], True)]),
    item("basic_algorithms", "hamming_distance", "takes strings left and right and returns the number of differing positions plus any length difference", "def solve(left, right):\n    return sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))", [tc(["karolin", "kathrin"], 3), tc(["", ""], 0), tc(["a", "b"], 1), tc(["abc", "ab"], 1), tc(["same", "same"], 0)]),
    item("basic_algorithms", "matrix_diagonal", "takes a square matrix and returns the sum of its main diagonal", "def solve(matrix):\n    return sum(matrix[index][index] for index in range(len(matrix)))", [tc([[[1, 2], [3, 4]]], 5), tc([[[7]]], 7), tc([[]], 0), tc([[[1, 0, 0], [0, 2, 0], [0, 0, 3]]], 6), tc([[[-1, 2], [3, -4]]], -5)]),
]

COMPOSITIONS = [
    item("compositions", "sum_unique_even", "takes values and returns the sum of distinct even values", "def solve(values):\n    return sum(value for value in set(values) if value % 2 == 0)", [tc([[1, 2, 2, 4]], 6), tc([[]], 0), tc([[3, 5]], 0), tc([[-2, -2, 1]], -2), tc([[0, 2, 4]], 6)]),
    item("compositions", "count_palindromic_words", "takes a sentence and returns how many whitespace-separated words are palindromes with case-sensitive characters", "def solve(text):\n    return sum(word == word[::-1] for word in text.split())", [tc(["level noon test"], 2), tc([""], 0), tc(["a abba"], 2), tc(["Python"], 0), tc(["aa bb cc"], 3)]),
    item("compositions", "largest_odd", "takes values and returns the largest odd value, or None when no odd value exists", "def solve(values):\n    odds = [value for value in values if value % 2]\n    return max(odds) if odds else None", [tc([[2, 7, 4]], 7), tc([[]], None), tc([[2, 4]], None), tc([[-3, 5, 1]], 5), tc([[0, -1]], -1)]),
    item("compositions", "count_nonspace_characters", "takes text and returns the number of characters other than whitespace", "def solve(text):\n    return sum(not character.isspace() for character in text)", [tc(["a b"], 2), tc([""], 0), tc(["  hi\n"], 2), tc(["Python"], 6), tc(["a\tb"], 2)]),
    item("compositions", "common_sorted_unique", "takes lists left and right and returns their common values once each in ascending order", "def solve(left, right):\n    return sorted(set(left) & set(right))", [tc([[3, 1, 3], [2, 3]], [3]), tc([[], [1]], []), tc([[1, 2], [2, 1]], [1, 2]), tc([[-1, 0], [-1]], [-1]), tc([[5], [4]], [])]),
    item("compositions", "sum_absolute_digits", "takes a list of integers and returns the sum of all decimal digits of their absolute values", "def solve(values):\n    return sum(sum(int(digit) for digit in str(abs(value))) for value in values)", [tc([[12, -3]], 6), tc([[]], 0), tc([[0, 5]], 5), tc([[-99]], 18), tc([[10, 20]], 3)]),
    item("compositions", "reverse_each_word", "takes a sentence and returns the words in the same order with each word's characters reversed", "def solve(text):\n    return ' '.join(word[::-1] for word in text.split())", [tc(["one two"], "eno owt"), tc([""], ""), tc(["hello"], "olleh"), tc(["a bb ccc"], "a bb ccc"), tc(["Gen Py"], "neG yP")]),
    item("compositions", "all_prime", "takes values and returns True only when every value is prime; an empty list returns True", "def solve(values):\n    def prime(n):\n        if n < 2:\n            return False\n        divisor = 2\n        while divisor * divisor <= n:\n            if n % divisor == 0:\n                return False\n            divisor += 1\n        return True\n    return all(prime(value) for value in values)", [tc([[2, 3, 5]], True), tc([[]], True), tc([[2, 4]], False), tc([[17]], True), tc([[1, 2]], False)]),
    item("compositions", "merge_unique_sorted", "takes two sorted lists and returns their merged values with duplicates removed", "def solve(left, right):\n    return sorted(set(left) | set(right))", [tc([[1, 2], [2, 3]], [1, 2, 3]), tc([[], [1]], [1]), tc([[2, 2], [2]], [2]), tc([[-1, 4], [0, 4]], [-1, 0, 4]), tc([[5], []], [5])]),
    item("compositions", "count_even_above", "takes values and threshold and counts even values strictly greater than threshold", "def solve(values, threshold):\n    return sum(value % 2 == 0 and value > threshold for value in values)", [tc([[1, 2, 4, 5], 1], 2), tc([[], 0], 0), tc([[2, 4], 4], 0), tc([[-2, 0, 6], -1], 2), tc([[10], 10], 0)]),
]

OPENERS = [
    "Write a Python function named solve.", "Implement solve.", "Create solve as a small pure Python function.",
    "Define solve with the following behavior.", "Write solve so that it returns the requested result.", "Implement the function solve without printing anything.",
]
SPLIT_OPENERS = {
    "train": ["Write a Python function named solve.", "Implement solve.", "Create solve as a small pure Python function.", "Define solve with the following behavior.", "Write solve so that it returns the requested result.", "Implement the function solve without printing anything."],
    "validation": ["Please define a Python function named solve.", "How would you implement solve in Python?", "Provide a pure Python solve function.", "Complete the following Python behavior in solve.", "Write solve to satisfy this contract.", "Return a concise implementation of solve."],
    "challenge": ["Implement a pure Python function named solve.", "Write solve for the behavior below.", "Provide ordinary Python code defining solve.", "Create solve and return the specified value.", "Solve this programming task with a function named solve.", "Give a deterministic implementation of solve."],
}
CLARIFIERS = ["Check boundary values as well as ordinary values.", "Include the empty or zero case when it is defined.", "Preserve the stated ordering and value conventions.", "Use the exact return type described above.", "Treat negative values according to the contract.", "Keep comparisons case-sensitive unless the contract says otherwise.", "Do not print; return the result from solve.", "A direct, readable implementation is sufficient."]
QUALIFIERS = ["Prefer a short readable solution.", "Use ordinary Python data structures.", "The result should be deterministic for every valid input.", "Keep the behavior self-contained.", "Return only the computed value.", "Do not mutate caller-owned input unless explicitly required.", "The function should work for small examples and boundary cases.", "Use meaningful control flow rather than external helpers.", "Preserve the order specified by the contract.", "Make the implementation easy to inspect.", "Handle the defined empty case directly.", "No output other than the return value is needed."]
CONNECTORS = ["It should", "The function must", "Have it", "Make solve", "The required behavior is that it", "Return a value that"]
TAILS = ["Handle the stated edge cases exactly.", "Keep the input unchanged.", "Use deterministic behavior.", "Return the result directly.", "Do not use external resources."]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_for(skill, index, split, kind="normal"):
    opener = SPLIT_OPENERS.get(split, OPENERS)[index % len(SPLIT_OPENERS.get(split, OPENERS))]
    connector = CONNECTORS[(index // len(OPENERS)) % len(CONNECTORS)]
    tail = TAILS[(index // (len(OPENERS) * len(CONNECTORS))) % len(TAILS)]
    clarifier = CLARIFIERS[(index // (len(OPENERS) * len(CONNECTORS) * len(TAILS))) % len(CLARIFIERS)]
    qualifier = QUALIFIERS[(index // (len(OPENERS) * len(CONNECTORS) * len(TAILS) * len(CLARIFIERS))) % len(QUALIFIERS)]
    example = skill["cases"][index % len(skill["cases"])]
    example_text = json.dumps({"args": example["args"], "expected": example["expected"]}, ensure_ascii=False, sort_keys=True)
    description = skill["description"]
    return f"{opener} {connector.lower()} {description}. {tail} {clarifier} {qualifier} For example, an input case is {example_text}."


def make_record(split, skill, index, template_family, difficulty, seed, task_style="code_generation", kind="normal"):
    return {"id": f"v3_{split}_{index:05d}", "instruction": prompt_for(skill, index, split, kind), "input": "", "response": skill["code"], "response_type": "code", "category": skill["category"], "skill_id": skill["skill"], "prompt_template_id": f"{template_family}.{index % 6:02d}", "difficulty": difficulty, "task_style": task_style, "function_name": "solve", "test_cases": skill["cases"], "provenance": {"generator": GENERATOR, "dataset_version": VERSION, "seed": seed, "source": "independently authored semantic catalog", "training_excluded": split in {"challenge", "sanity"}, "hyperparameter_selection_excluded": split in {"challenge", "sanity"}, "optimizer_excluded": split in {"challenge", "sanity"}}}


def build_normal(split, count, seed):
    rng = random.Random(seed + {"train": 0, "validation": 100, "challenge": 200}[split])
    rows = []
    for skill_index, skill in enumerate(CATALOG):
        amount = count // len(CATALOG) + (skill_index < count % len(CATALOG))
        for _ in range(amount):
            index = len(rows)
            rows.append(make_record(split, skill, index, f"{split}.semantic.{skill['skill']}", "easy" if index % 20 < 13 else "medium", seed))
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["id"] = f"v3_{split}_{index:05d}"
        row["prompt_template_id"] = f"{split}.semantic.{row['skill_id']}.{index % 6:02d}"
        row["instruction"] = prompt_for(next(skill for skill in CATALOG if skill["skill"] == row["skill_id"]), index, split)
    return rows


def build_challenge(seed):
    rows = build_normal("challenge", 150, seed)
    rows = rows[:150]
    for index, row in enumerate(rows):
        row["prompt_template_id"] = f"challenge.natural.{row['skill_id']}.{index % 4:02d}"
        row["difficulty"] = "medium"
        row["provenance"]["challenge_type"] = "known_skill_rephrase"
    rng = random.Random(seed + 333)
    for offset, skill in enumerate(COMPOSITIONS):
        for local in range(5):
            index = 150 + offset * 5 + local
            row = make_record("challenge", skill, index, f"challenge.composition.{skill['skill']}", "medium", seed, kind="composition")
            row["id"] = f"v3_challenge_{index:05d}"
            row["prompt_template_id"] = f"challenge.composition.{skill['skill']}.{local:02d}"
            row["provenance"]["challenge_type"] = "composition"
            rows.append(row)
    rng.shuffle(rows)
    for index, row in enumerate(rows): row["id"] = f"v3_challenge_{index:05d}"
    return rows


def sanity_rows():
    selected = [
        ("Write a Python function named solve that returns True for an even integer and False for an odd integer.", "parity_even"),
        ("Write solve(text) to reverse a string.", "reverse_text"), ("Write solve(values) to return the largest list item.", "maximum_item"),
        ("Write solve(text) to test whether a string is a palindrome.", "palindrome"), ("Write solve(n) to calculate factorial.", "factorial"),
        ("Write solve(values) to remove duplicate list values while preserving order.", "stable_unique"), ("Write solve(text) to count vowels.", "vowel_count"),
        ("Write solve(values) to sort a list using insertion sort.", "insertion_sort"), ("Write solve(n) to return the nth Fibonacci number.", "fibonacci"),
        ("Write solve(a, b) to return the greatest common divisor.", "gcd"), ("Write solve(n) to determine whether n is prime.", "prime_check"),
        ("Write solve(values) to return the sum of a list.", "sum_items"), ("Write solve(n) to return the sum of its digits.", "digit_sum"),
        ("Write solve(text) to count whitespace-separated words.", "word_count"), ("Write solve(left, right) to determine whether two strings are anagrams.", "anagram"),
        ("Write solve(values) to return running totals.", "running_totals"), ("Write solve(text) to check balanced brackets.", "balanced_brackets"),
        ("Write solve(values, target) to search a sorted list with binary search.", "binary_search"), ("Write solve(text) to count alphabetic characters.", "character_histogram"),
        ("Write solve(left, right) to merge two sorted lists.", "merge_sorted"),
    ]
    rows = []
    for index, (instruction, skill_name) in enumerate(selected):
        skill = next(skill for skill in CATALOG if skill["skill"] == skill_name)
        rows.append({"id": f"v3_sanity_{index:02d}", "instruction": instruction, "input": "", "response": skill["code"], "response_type": "code", "category": "sanity", "skill_id": f"sanity.{skill_name}", "prompt_template_id": f"sanity.canonical.{index:02d}", "difficulty": "easy", "task_style": "evaluation_only", "function_name": "solve", "test_cases": skill["cases"], "provenance": {"generator": GENERATOR, "dataset_version": VERSION, "seed": SEED, "training_excluded": True, "validation_excluded": True, "hyperparameter_selection_excluded": True, "optimizer_excluded": True}})
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="data/instruction/python_v3"); args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    train, validation, challenge, sanity = build_normal("train", 3000, SEED), build_normal("validation", 300, SEED), build_challenge(SEED), sanity_rows()
    rows_by_name = {"train": train, "validation": validation, "challenge": challenge, "sanity": sanity}
    for name, rows in rows_by_name.items(): write_jsonl(output / f"{name}.jsonl", rows)
    manifest = {"dataset_name": "GenPy-SFT-v3-Semantic", "dataset_version": VERSION, "generator_version": GENERATOR, "seed": SEED, "external_data_used": False, "internet_used": False, "independently_authored": True, "sft_training_splits": ["train", "validation"], "challenge_training_excluded": True, "sanity_training_excluded": True, "counts": {name: len(rows) for name, rows in rows_by_name.items()}, "files": {name: {"sha256": sha256(output / f"{name}.jsonl"), "count": len(rows)} for name, rows in rows_by_name.items()}, "core_skill_count": len(CATALOG), "challenge_composition_count": len(COMPOSITIONS), "function_interface": "def solve(...)", "tokenization_performed": False, "training_performed": False}
    (output / "DATASET_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "core_skills": len(CATALOG), "challenge_compositions": len(COMPOSITIONS), "hashes": {name: value["sha256"] for name, value in manifest["files"].items()}}, indent=2))


if __name__ == "__main__": main()
