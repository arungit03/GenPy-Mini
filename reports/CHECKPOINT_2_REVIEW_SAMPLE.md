# GenPy Checkpoint 2 Review Sample

This deterministic smoke sample uses seed 42. Production dataset population is pending; all 7 processed fixture examples are shown.

## 1. reverse-1

- Category: `strings`
- Task: `code_generation`
- Family: `reverse_string`
- Source: `fixture`
- Quality: `1.0`
- Syntax: PASS
- Instruction: Reverse a string in Python.
- Response: `text = input()` followed by `print(text[::-1])`

## 2. code-1

- Category: `linked_lists`
- Task: `code`
- Family: `linked_lists`
- Source: `fixture`
- Quality: `1.0`
- Syntax: PASS
- Code: `class Node` with an initializer storing `self.value`

## 3. odd-1

- Category: `conditions`
- Task: `code_generation`
- Family: `odd_even`
- Source: `fixture`
- Quality: `1.0`
- Syntax: PASS
- Instruction: Check if 10 is even.
- Response: Computes `number % 2 == 0` and prints the result.

## 4. odd-2

- Category: `conditions`
- Task: `code_generation`
- Family: `odd_even`
- Source: `fixture`
- Quality: `1.0`
- Syntax: PASS
- Instruction: Write a program to determine whether an integer is odd or even.
- Response: Reads an integer and prints `Even` or `Odd`.

## 5. odd-3

- Category: `conditions`
- Task: `code_generation`
- Family: `odd_even`
- Source: `fixture`
- Quality: `1.0`
- Syntax: PASS
- Instruction: Python odd/even checker.
- Response: Reads an integer and prints `Even` or `Odd`.

## 6. func-1

- Category: `functions`
- Task: `code_generation`
- Family: `family_1ba5f06d849e0dc7`
- Source: `fixture`
- Quality: `1.0`
- Syntax: PASS
- Instruction: Define a function that adds two numbers.
- Response: Defines `add(a, b)` returning `a + b`.

## 7. list-1

- Category: `lists`
- Task: `code_generation`
- Family: `family_4831756acfda4ee5`
- Source: `fixture`
- Quality: `0.8`
- Syntax: PASS
- Instruction: Remove duplicates from a list.
- Response: Uses `dict.fromkeys` to retain unique values.
