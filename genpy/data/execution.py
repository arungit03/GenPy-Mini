"""Controlled execution helpers for trusted generated tasks."""

from dataclasses import dataclass
import contextlib
import io
import json
import subprocess
import sys
import warnings
from typing import Any


@dataclass
class ExecutionResult:
    tested: bool
    passed: bool
    test_count: int
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


def _equal(actual: Any, expected: Any) -> bool:
    try:
        import numpy as np
        if isinstance(actual, np.ndarray):
            return bool(np.array_equal(actual, expected))
        if isinstance(actual, (float, np.floating)) or isinstance(expected, (float, np.floating)):
            return bool(np.isclose(actual, expected))
    except (ImportError, TypeError, ValueError):
        pass
    if hasattr(actual, "to_dict"):
        actual = actual.to_dict()
    return actual == expected


def execute_function_tests(code: str, function_name: str, test_cases: list[dict[str, Any]]) -> ExecutionResult:
    """Execute trusted generated code in an isolated namespace and run cases."""
    namespace: dict[str, Any] = {"__name__": "__generated_task__"}
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            exec(compile(code, "<generated>", "exec"), namespace, namespace)
            function = namespace[function_name]
            for case in test_cases:
                actual = function(*case.get("args", []), **case.get("kwargs", {}))
                if not _equal(actual, case.get("expected")):
                    return ExecutionResult(True, False, len(test_cases), output.getvalue(), error="semantic test mismatch")
        return ExecutionResult(True, True, len(test_cases), output.getvalue())
    except Exception as exc:  # generated code is rejected by the caller, never retained
        return ExecutionResult(True, False, len(test_cases), output.getvalue(), error=f"{type(exc).__name__}: {exc}")


def run_python_script(code: str, input_text: str = "", timeout_seconds: float = 3.0) -> ExecutionResult:
    """Run a script through a subprocess with captured output and a timeout."""
    payload = json.dumps(code)
    command = [sys.executable, "-c", f"import json; exec(compile(json.loads({payload!r}), '<dataset>', 'exec'))"]
    try:
        completed = subprocess.run(command, input=input_text, text=True, capture_output=True, timeout=timeout_seconds)
        return ExecutionResult(True, completed.returncode == 0, 1, completed.stdout, completed.stderr, None if completed.returncode == 0 else f"exit {completed.returncode}")
    except subprocess.TimeoutExpired as exc:
        return ExecutionResult(True, False, 1, exc.stdout or "", exc.stderr or "", "timeout")
