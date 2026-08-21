"""Deterministic coding-output extraction and safety-aware classification."""

from __future__ import annotations

import ast
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_CODE_START = re.compile(r"(?m)^(?:from\s+\w+|import\s+\w+|def\s+\w+|class\s+\w+|if\s+__name__\s*==|if\s+|for\s+|while\s+|print\s*\(|[A-Za-z_]\w*\s*=)")
_RISKY_IMPORTS = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests", "urllib", "http", "ftplib"}
_SPACE = re.compile(r"\s+")


def extract_python(text: str) -> str | None:
    fenced = _FENCE.findall(text)
    if fenced:
        return max(fenced, key=len).strip()
    match = _CODE_START.search(text)
    return text[match.start():].strip() if match else None


def normalized_output(text: str) -> str:
    return _SPACE.sub(" ", text.strip().lower())


def _safe_for_execution(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] in _RISKY_IMPORTS for alias in node.names):
            return False
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in _RISKY_IMPORTS:
            return False
    return True


def run_function_tests_subprocess(code: str, function_name: str, test_cases: list[dict], timeout_seconds: float = 2.0) -> dict:
    if not _safe_for_execution(code):
        return {"applicable": False, "passed": False, "category": "unsafe_for_execution"}
    payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
    cases = json.dumps(test_cases, ensure_ascii=False)
    script = """import base64, json\ncode = base64.b64decode(%r).decode('utf-8')\ncases = json.loads(%r)\nns = {'__name__': '__generated__'}\nexec(compile(code, '<generated>', 'exec'), ns, ns)\nfn = ns[%r]\nfor case in cases:\n    actual = fn(*case.get('args', []), **case.get('kwargs', {}))\n    if actual != case.get('expected'):\n        raise AssertionError('wrong result')\n""" % (payload, cases, function_name)
    with tempfile.TemporaryDirectory(prefix="genpy-eval-") as temp_dir:
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"}
        try:
            completed = subprocess.run([sys.executable, "-c", script], cwd=temp_dir, env=env, capture_output=True, text=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return {"applicable": True, "passed": False, "category": "timeout"}
    if completed.returncode != 0:
        return {"applicable": True, "passed": False, "category": "runtime_failure"}
    return {"applicable": True, "passed": True, "category": "functional_pass"}


def evaluate_code(generated_text: str, record: dict | None = None) -> dict:
    code = extract_python(generated_text)
    result = {"python_extracted": code is not None, "syntax_valid": False, "compile_valid": False, "executable": None, "functional_correct": None, "failure_category": None}
    if code is None:
        result["failure_category"] = "no_python_extracted"
        return result
    try:
        ast.parse(code)
        result["syntax_valid"] = True
        compile(code, "<generated>", "exec")
        result["compile_valid"] = True
    except SyntaxError:
        result["failure_category"] = "syntax_failure"
        return result
    tests = (record or {}).get("test_cases") or (record or {}).get("tests")
    function_name = (record or {}).get("function_name", "solve")
    if tests:
        execution = run_function_tests_subprocess(code, function_name, tests)
        result["executable"] = execution["passed"]
        result["functional_correct"] = execution["passed"]
        result["failure_category"] = None if execution["passed"] else execution["category"]
    return result


def generation_summary(outputs: list[str]) -> dict:
    normalized = [normalized_output(value) for value in outputs]
    counts = Counter(normalized)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    return {"count": len(outputs), "unique_output_rate": len(counts) / len(outputs) if outputs else None, "exact_duplicate_generations": duplicate_count, "repeated_output_rate": duplicate_count / len(outputs) if outputs else None, "normalized_duplicate_generations": duplicate_count, "response_prefix_collapse_warning": len({value[:80] for value in normalized}) <= 1 if normalized else False}
