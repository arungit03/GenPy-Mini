"""Stable exact and conservative near-duplicate detection."""

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import re

from .normalize import content_text
from .schema import CodeExample, InstructionExample


@dataclass
class DeduplicationReport:
    exact_duplicates: int = 0
    instruction_duplicates: int = 0
    code_duplicates: int = 0
    near_duplicates: int = 0
    examples_removed: int = 0


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _instruction(example: InstructionExample | CodeExample) -> str:
    return _norm(getattr(example, "instruction", ""))


def _code(example: InstructionExample | CodeExample) -> str:
    return _norm(getattr(example, "response", getattr(example, "code", "")))


def _shingles(value: str, size: int = 3) -> set[str]:
    tokens = value.split()
    if len(tokens) <= size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def deduplicate_examples(
    examples: list[InstructionExample | CodeExample],
    near_duplicate: bool = True,
    near_duplicate_threshold: float = 0.90,
    near_duplicate_across_families: bool = True,
) -> tuple[list[InstructionExample | CodeExample], DeduplicationReport]:
    """Keep stable occurrences, optionally respecting explicit family boundaries."""
    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    report = DeduplicationReport()
    kept: list[InstructionExample | CodeExample] = []
    exact_seen: set[str] = set()
    instruction_seen: set[str] = set()
    code_seen: set[str] = set()
    shingle_index: defaultdict[str, list[int]] = defaultdict(list)
    kept_shingles: list[set[str]] = []
    kept_text: list[str] = []
    for example in examples:
        exact_key = _hash(f"{_instruction(example)}\n{_code(example)}")
        instruction_key = _hash(_instruction(example))
        code_key = _hash(_code(example))
        if exact_key in exact_seen:
            report.exact_duplicates += 1
            report.examples_removed += 1
            continue
        if instruction_key in instruction_seen:
            report.instruction_duplicates += 1
        if code_key in code_seen:
            report.code_duplicates += 1
        shingles = _shingles(_norm(content_text(example)))
        near_hit = False
        if near_duplicate and shingles:
            candidates: set[int] = set()
            for shingle in shingles:
                candidates.update(shingle_index[shingle][:100])
            current_text = _norm(content_text(example))
            for index in sorted(candidates):
                if not near_duplicate_across_families:
                    current_family = getattr(example, "family_id", "")
                    previous_family = getattr(kept[index], "family_id", "")
                    if current_family and previous_family and current_family != previous_family:
                        continue
                shingle_similarity = _jaccard(shingles, kept_shingles[index])
                character_similarity = SequenceMatcher(None, current_text, kept_text[index]).ratio()
                if max(shingle_similarity, character_similarity) >= near_duplicate_threshold:
                    near_hit = True
                    break
        if near_hit:
            report.near_duplicates += 1
            report.examples_removed += 1
            continue
        index = len(kept)
        kept.append(example)
        exact_seen.add(exact_key)
        instruction_seen.add(instruction_key)
        code_seen.add(code_key)
        kept_shingles.append(shingles)
        kept_text.append(_norm(content_text(example)))
        for shingle in shingles:
            shingle_index[shingle].append(index)
    return kept, report
