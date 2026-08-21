"""Central registries and benchmark-exclusion support."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

TASK_TYPES = {
    "code_generation", "code_completion", "bug_fixing", "code_explanation",
    "code_optimization", "algorithm_implementation", "data_structure",
    "library_usage", "code",
}
CATEGORIES = {
    "beginner", "conditions", "loops", "functions", "recursion", "strings", "lists",
    "tuples", "sets", "dictionaries", "oop", "files", "exceptions", "modules",
    "algorithms", "sorting", "searching", "arrays", "linked_lists", "stacks", "queues",
    "trees", "graphs", "dynamic_programming", "greedy", "backtracking", "math", "numpy",
    "pandas", "matplotlib", "regex", "debugging", "optimization", "code_completion",
    "intermediate_python", "misc",
}
CATEGORY_ALIASES = {
    "linked-list": "linked_lists", "linked list": "linked_lists", "linkedlist": "linked_lists",
    "linked lists": "linked_lists", "dict": "dictionaries", "dictionary": "dictionaries",
    "oop_python": "oop", "object_oriented": "oop", "code completion": "code_completion",
    "dynamic programming": "dynamic_programming", "data structures": "data_structure",
}


def normalize_category(value: str) -> str:
    """Return a canonical category spelling or raise for unknown categories."""
    if not isinstance(value, str):
        raise ValueError("category must be a string")
    key = value.strip().lower().replace("/", "_")
    key = CATEGORY_ALIASES.get(key, key.replace(" ", "_"))
    if key not in CATEGORIES:
        raise ValueError(f"Unknown Python dataset category: {value!r}")
    return key


def normalize_task_type(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("task_type must be a string")
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {value!r}")
    return key


@dataclass
class ExclusionRegistry:
    """Deterministic blacklist for benchmark leakage and known bad records."""

    ids: set[str] = field(default_factory=set)
    instruction_hashes: set[str] = field(default_factory=set)
    content_hashes: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def is_excluded(self, example: Any) -> bool:
        instruction = getattr(example, "instruction", "")
        response = getattr(example, "response", getattr(example, "code", ""))
        instruction_hash = hashlib.sha256(instruction.strip().lower().encode("utf-8")).hexdigest()
        content_hash = hashlib.sha256(f"{instruction}\n{response}".encode("utf-8")).hexdigest()
        return (getattr(example, "id", "") in self.ids
                or instruction_hash in self.instruction_hashes
                or content_hash in self.content_hashes)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExclusionRegistry":
        path = Path(path)
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            ids=set(raw.get("problem_ids", raw.get("ids", []))),
            instruction_hashes=set(raw.get("instruction_hashes", [])),
            content_hashes=set(raw.get("content_hashes", raw.get("hashes", []))),
            notes=list(raw.get("notes", [])),
        )
