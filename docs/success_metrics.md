# Success Metrics

GenPy quality must be measured with a private, frozen evaluation set created in a later
phase. Evaluation prompts must never appear in training data. Comparisons across GenPy-5M,
GenPy-25M, and GenPy-100M must use the same prompts, decoding settings, tests, and
reporting format.

Targets below are project goals for GenPy-100M, not guaranteed results.

## Initial GenPy-100M Targets

- Python syntax-validity rate: at least 85%
- Code-only format accuracy: at least 90%
- Unit-test pass rate on beginner problems: at least 50%
- Unsafe-code generation rate: below 2%
- No known training/test data leakage
- GenPy-100M must perform better than the GenPy-25M checkpoint on the same frozen
  evaluation set

## Metric Definitions

### Validation Loss

Measure next-token cross-entropy on a held-out validation split that is separate from the
training set. Track loss for each checkpoint, but do not judge quality by loss alone.

### Python Syntax-Validity Rate

Run `ast.parse()` on generated code after removing only the model's special end token.
Report the percentage of generations that parse successfully.

### Compilation-Success Rate

Run Python `compile()` on generated code in validation tooling without executing it.
Report the percentage that compile successfully.

### Unit-Test Pass Rate

For prompts with reference tests, execute generated code inside a restricted future
sandbox and report the percentage that passes all tests.

### pass@1

Generate one answer per prompt using fixed decoding settings. A prompt counts as passed
when that first generation satisfies the problem's tests or acceptance criteria.

### Code-Only Output-Format Accuracy

Check whether the output contains Python code only, without Markdown fences or explanatory
text, unless the prompt explicitly asks for explanation.

### Prompt Relevance

Score whether the generated program addresses the requested task. This may use rubric
labels or targeted tests depending on the prompt type.

### Duplicate or Memorized Output Rate

Compare generated outputs against training examples and near-duplicate fingerprints.
Report suspected memorized or duplicated generations.

### Average Generation Time

Measure wall-clock time per prompt on a named hardware and software environment with
fixed decoding settings.

### Peak Inference Memory

Measure peak memory during generation. On CUDA, use PyTorch CUDA memory reporting. On CPU,
record process memory using platform-appropriate tooling.

### Unsafe-Code Generation Rate

Scan generated code for unsafe operations such as unsandboxed filesystem deletion,
network access, process spawning, credential access, or attempts to escape restrictions.
Report the percentage of generations flagged by the safety checks.

## Evaluation Requirements

- A private, frozen evaluation set must be created later.
- Evaluation problems must never appear in training data.
- The same prompts, decoding settings, and tests must be used when comparing 5M, 25M,
  and 100M models.
- Test results must include both successful and failed examples.
- Model quality must be judged using execution tests, not only training loss.
- Failures must be saved with prompts, generations, error types, and test output.
