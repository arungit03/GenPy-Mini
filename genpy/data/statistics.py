"""Character/line-level dataset statistics; tokenization is deferred."""

from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any

from .normalize import content_text
from .schema import CodeExample, InstructionExample


@dataclass
class DatasetStatistics:
    total_examples: int
    category_counts: dict[str, int]
    task_type_counts: dict[str, int]
    source_counts: dict[str, int]
    average_instruction_characters: float
    average_response_characters: float
    minimum_response_length: int
    maximum_response_length: int
    syntax_valid_percentage: float
    quality_score_distribution: dict[str, int]
    examples_rejected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_statistics(examples: list[InstructionExample | CodeExample], examples_rejected: int = 0) -> DatasetStatistics:
    total = len(examples)
    instruction_lengths = [len(getattr(example, "instruction", "")) for example in examples]
    response_lengths = [len(getattr(example, "response", getattr(example, "code", ""))) for example in examples]
    syntax_count = sum(getattr(example, "syntax_valid", False) is True for example in examples)
    distribution = Counter("1.0" if example.quality_score >= 1.0 else "0.8" if example.quality_score >= 0.8 else "0.5_or_lower" for example in examples)
    return DatasetStatistics(
        total_examples=total,
        category_counts=dict(Counter(example.category for example in examples)),
        task_type_counts=dict(Counter(example.task_type for example in examples)),
        source_counts=dict(Counter(example.source for example in examples)),
        average_instruction_characters=sum(instruction_lengths) / total if total else 0.0,
        average_response_characters=sum(response_lengths) / total if total else 0.0,
        minimum_response_length=min(response_lengths, default=0),
        maximum_response_length=max(response_lengths, default=0),
        syntax_valid_percentage=(100.0 * syntax_count / total) if total else 0.0,
        quality_score_distribution=dict(distribution),
        examples_rejected=examples_rejected,
    )


def classify_dataset_size(count: int) -> str:
    if count < 25_000:
        return "prototype_only"
    if count < 50_000:
        return "small_training_set"
    if count < 90_000:
        return "usable_training_set"
    if count <= 110_000:
        return "production_candidate_target_met"
    return "above_target"
