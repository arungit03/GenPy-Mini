"""Shared types for programmatic task generators."""

from dataclasses import dataclass
from typing import Any

from ..schema import InstructionExample


@dataclass
class GeneratedTask:
    example: InstructionExample
    function_name: str
    test_cases: list[dict[str, Any]]


def difficulty_for(index: int) -> str:
    slot = index % 20
    if slot < 9:
        return "easy"
    if slot < 17:
        return "medium"
    return "hard"
