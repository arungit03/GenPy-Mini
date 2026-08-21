"""Deterministic family-grouped train/validation/test splitting."""

from collections import defaultdict
import random
from typing import TypeVar

from .normalize import derive_family_id
from .schema import CodeExample, InstructionExample

Example = TypeVar("Example", InstructionExample, CodeExample)


def split_examples(
    examples: list[Example],
    train_ratio: float = 0.90,
    validation_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, list[Example]]:
    """Assign whole family groups to deterministic splits."""
    if min(train_ratio, validation_ratio, test_ratio) < 0 or abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("split ratios must be non-negative and sum to 1.0")
    groups: defaultdict[str, list[Example]] = defaultdict(list)
    for example in examples:
        family = getattr(example, "family_id", "") or derive_family_id(getattr(example, "instruction", ""))
        example.family_id = family
        groups[family].append(example)
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    targets = {
        "train": len(examples) * train_ratio,
        "validation": len(examples) * validation_ratio,
        "test": len(examples) * test_ratio,
    }
    result: dict[str, list[Example]] = {"train": [], "validation": [], "test": []}
    # Tiny grouped datasets otherwise commonly put every family in train because
    # a single family can be larger than the validation/test targets. Reserve
    # the two smallest groups when there are enough independent families.
    if len(group_items) >= 3:
        reserved = sorted(group_items, key=lambda item: (len(item[1]), item[0]))
        validation_family, validation_members = reserved[0]
        test_family, test_members = reserved[1]
        result["validation"].extend(validation_members)
        result["test"].extend(test_members)
        reserved_names = {validation_family, test_family}
        group_items = [(family, members) for family, members in group_items if family not in reserved_names]
    for family, members in group_items:
        deficits = {name: targets[name] - len(result[name]) for name in result}
        split_name = max(deficits, key=lambda name: (deficits[name], {"train": 2, "validation": 1, "test": 0}[name]))
        result[split_name].extend(members)
    return result


def family_overlap(splits: dict[str, list[Example]]) -> set[str]:
    seen: dict[str, str] = {}
    overlap: set[str] = set()
    for split_name, examples in splits.items():
        for example in examples:
            family = getattr(example, "family_id", "")
            if family in seen and seen[family] != split_name:
                overlap.add(family)
            seen[family] = split_name
    return overlap
