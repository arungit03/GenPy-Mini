# GenPy V1 Scope

GenPy V1 is a small Python-focused code-generation model. It should translate beginner
and intermediate natural-language programming requests into readable Python programs.

## Supported Problem Categories

- Basic input and output
- Variables and data types
- Arithmetic programs
- Conditions
- Loops
- Number-based problems
- Strings
- Lists, tuples, sets, and dictionaries
- Functions
- Basic recursion
- Basic exception handling
- Simple file-handling programs
- Beginner object-oriented programming
- Simple algorithms such as searching and sorting
- Simple debugging and correction of Python code

## Example Supported Requests

- Check whether a number is odd or even
- Find the factorial of a number
- Check whether a number is an Armstrong number
- Reverse a string
- Find the largest number in a list
- Create a calculator using functions
- Create a basic Python class
- Fix a simple syntax or logic error

## Default Output Behavior

- Return Python code only unless the user explicitly requests an explanation.
- Generate readable beginner-friendly code.
- Use meaningful variable names.
- Avoid unnecessary libraries.
- Include input handling when required by the prompt.
- Never include Markdown code fences in raw model output unless requested.
- Finish generation using a special end token.

## V1 Limitations

- GenPy V1 will not compete with large commercial LLMs.
- It may fail on advanced algorithms and large projects.
- It will not initially support full-stack application generation.
- It will not initially support advanced frameworks such as Django, PyTorch, or TensorFlow.
- It will not guarantee that every generated program is correct.
- It will not execute generated code directly on the user's main operating system.
- Future code execution must use an isolated environment with network, time, and memory
  restrictions.

## Possible V2 Features

- Broader algorithm coverage
- Better debugging and explanation modes
- Safer sandboxed execution for generated code
- Small project generation across multiple files
- Introductory web, data, and automation scripts
- Improved prompt-following and refusal behavior for unsafe requests
