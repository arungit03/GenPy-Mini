"""Validated loader for the Checkpoint 2 dataset YAML configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


def _bool(section: Mapping[str, Any], name: str) -> bool:
    value = section.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"dataset.{name} must be a boolean")
    return value


@dataclass(frozen=True)
class DeduplicationConfig:
    exact: bool
    instruction: bool
    code: bool
    near_duplicate: bool
    near_duplicate_threshold: float


@dataclass(frozen=True)
class QualityConfig:
    minimum_score: float


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    language: str
    target_examples: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    seed: int
    minimum_instruction_chars: int
    minimum_response_chars: int
    strict_python_syntax: bool
    deduplication: DeduplicationConfig
    quality: QualityConfig


def load_data_config(path: str | Path) -> DatasetConfig:
    """Load and validate the complete dataset pipeline configuration."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("dataset"), Mapping):
        raise ValueError("Dataset configuration must contain a 'dataset' mapping")
    section = raw["dataset"]
    required = {
        "name", "language", "target_examples", "train_ratio", "validation_ratio", "test_ratio",
        "seed", "minimum_instruction_chars", "minimum_response_chars", "strict_python_syntax",
        "deduplication", "quality",
    }
    unknown = sorted(set(section) - required)
    missing = sorted(required - set(section))
    if unknown:
        raise ValueError(f"Unknown dataset configuration field(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"Missing dataset configuration field(s): {', '.join(missing)}")
    if not isinstance(section["name"], str) or not section["name"].strip():
        raise ValueError("dataset.name must be a non-empty string")
    if section["language"] != "python":
        raise ValueError("dataset.language must be python")
    for name in ("target_examples", "seed", "minimum_instruction_chars", "minimum_response_chars"):
        value = section[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"dataset.{name} must be a non-negative integer")
    if section["target_examples"] <= 0 or section["minimum_instruction_chars"] <= 0 or section["minimum_response_chars"] <= 0:
        raise ValueError("dataset target and minimum character values must be positive")
    ratios = [section[name] for name in ("train_ratio", "validation_ratio", "test_ratio")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in ratios):
        raise ValueError("dataset split ratios must be non-negative numbers")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("dataset split ratios must sum to 1.0")
    dedup = section["deduplication"]
    quality = section["quality"]
    if not isinstance(dedup, Mapping) or not isinstance(quality, Mapping):
        raise ValueError("dataset.deduplication and dataset.quality must be mappings")
    dedup_required = {"exact", "instruction", "code", "near_duplicate", "near_duplicate_threshold"}
    if set(dedup) != dedup_required:
        raise ValueError("dataset.deduplication has missing or unknown fields")
    threshold = dedup["near_duplicate_threshold"]
    if any(not isinstance(dedup[name], bool) for name in ("exact", "instruction", "code", "near_duplicate")):
        raise ValueError("dataset deduplication switches must be booleans")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 < threshold <= 1.0:
        raise ValueError("dataset.near_duplicate_threshold must be in (0, 1]")
    if set(quality) != {"minimum_score"}:
        raise ValueError("dataset.quality must contain only minimum_score")
    minimum_score = quality["minimum_score"]
    if isinstance(minimum_score, bool) or not isinstance(minimum_score, (int, float)) or not 0.0 <= minimum_score <= 1.0:
        raise ValueError("dataset.quality.minimum_score must be between 0 and 1")
    return DatasetConfig(
        name=section["name"], language=section["language"], target_examples=section["target_examples"],
        train_ratio=float(section["train_ratio"]), validation_ratio=float(section["validation_ratio"]),
        test_ratio=float(section["test_ratio"]), seed=section["seed"],
        minimum_instruction_chars=section["minimum_instruction_chars"],
        minimum_response_chars=section["minimum_response_chars"],
        strict_python_syntax=_bool(section, "strict_python_syntax"),
        deduplication=DeduplicationConfig(
            exact=dedup["exact"], instruction=dedup["instruction"], code=dedup["code"],
            near_duplicate=dedup["near_duplicate"], near_duplicate_threshold=float(threshold),
        ),
        quality=QualityConfig(minimum_score=float(minimum_score)),
    )
