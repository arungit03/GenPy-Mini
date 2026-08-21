"""Conservative, semantics-preserving normalization for Python records."""

import hashlib
import re

from .registry import normalize_category, normalize_task_type
from .schema import CodeExample, InstructionExample

_FENCE = re.compile(r"^\s*```(?:python|py)?\s*\n(?P<code>[\s\S]*?)\n?```\s*$", re.IGNORECASE)


def clean_text(value: str) -> str:
    """Normalize line endings/null bytes and remove outer blank space only."""
    return value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def strip_single_python_fence(value: str) -> str:
    match = _FENCE.match(value)
    return match.group("code").strip() if match else value


def derive_family_id(instruction: str, explicit: str = "") -> str:
    """Use explicit family metadata or a conservative deterministic fingerprint."""
    if explicit and explicit.strip():
        return re.sub(r"[^a-z0-9_]+", "_", explicit.lower().strip()).strip("_")
    text = re.sub(r"[^a-z0-9 ]+", " ", instruction.lower())
    text = re.sub(r"\s+", " ", text).strip()
    families = {
        "prime_check": ("prime",),
        "palindrome": ("palindrome",), "reverse_string": ("reverse", "string"),
        "fibonacci": ("fibonacci",), "factorial": ("factorial",),
        "sorting": ("sort",), "binary_search": ("binary", "search"),
    }
    if re.search(r"\b(?:odd|even)\b", text):
        return "odd_even"
    for family, terms in families.items():
        if all(term in text for term in terms):
            return family
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"family_{fingerprint}"


def _stable_id(task_type: str, instruction: str, input_text: str, response: str) -> str:
    payload = f"{task_type}\n{instruction}\n{input_text}\n{response}".encode("utf-8")
    return "py_" + hashlib.sha256(payload).hexdigest()[:20]


def normalize_example(example: InstructionExample | CodeExample) -> InstructionExample | CodeExample:
    """Normalize text and labels without changing Python indentation."""
    if isinstance(example, CodeExample):
        example.code = strip_single_python_fence(clean_text(example.code))
        example.category = normalize_category(example.category)
        example.task_type = normalize_task_type(example.task_type)
        example.language = "python"
        example.source = clean_text(example.source) or "unknown"
        example.family_id = derive_family_id("", example.family_id or example.category)
        if not example.id.strip():
            example.id = _stable_id(example.task_type, "", "", example.code)
        return example
    example.instruction = clean_text(example.instruction)
    example.input = clean_text(example.input)
    example.response = strip_single_python_fence(clean_text(example.response))
    example.category = normalize_category(example.category)
    example.task_type = normalize_task_type(example.task_type)
    example.language = "python"
    example.source = clean_text(example.source) or "unknown"
    example.family_id = derive_family_id(example.instruction, example.family_id)
    if not example.id.strip():
        example.id = _stable_id(example.task_type, example.instruction, example.input, example.response)
    return example


def content_text(example: InstructionExample | CodeExample) -> str:
    if isinstance(example, CodeExample):
        return example.code
    return f"{example.instruction}\n{example.input}\n{example.response}"
