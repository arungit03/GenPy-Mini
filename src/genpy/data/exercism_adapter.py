"""Deterministic Exercism instruction-answer pairing adapter.

Each exercises/practice/<slug>/ directory in the pinned Exercism Python track pairs
.docs/instructions.md (prompt) with .meta/example.py (canonical solution) by slug. This
module builds one InstructionRecord per exercise, reusing the same normalization, secret/
PII/unsafe scanning, quality scoring, exact/near deduplication, and group-aware splitting
used for pretraining data. No LLM is used; solutions are Exercism's own maintainer-authored
canonical examples.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genpy.data.near_dedup import jaccard_similarity, token_shingles
from genpy.data.normalize import NormalizationError, normalize_python_bytes
from genpy.data.pii_scan import scan_pii
from genpy.data.python_validation import validate_python
from genpy.data.quality import score_quality
from genpy.data.safety_filter import scan_unsafe_content
from genpy.data.schemas import InstructionRecord, Message, content_sha256, stable_record_id
from genpy.data.secret_scan import scan_secrets
from genpy.data.source_registry import SourceEntry

SYSTEM_MESSAGE = "You are GenPy, a Python code generator."
DIFFICULTY_BUCKETS = {
    1: "beginner",
    2: "beginner",
    3: "beginner",
    4: "intermediate",
    5: "intermediate",
    6: "intermediate",
    7: "advanced",
    8: "advanced",
    9: "advanced",
}


@dataclass(frozen=True, slots=True)
class ExercismRejection:
    """One skipped exercise and the machine-readable reason."""

    slug: str
    reason: str


def _extracted_root(source: SourceEntry, raw_root: Path) -> Path:
    source_root = raw_root / source.id / source.version
    candidates = [item for item in source_root.iterdir() if item.is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one extracted root under {source_root}")
    return candidates[0]


def iter_exercise_pairs(
    source: SourceEntry,
    raw_root: Path,
    normalization: dict[str, Any],
    quality_config: dict[str, Any],
) -> Iterator[InstructionRecord | ExercismRejection]:
    """Yield one InstructionRecord (or a rejection) per deterministically paired exercise."""
    root = _extracted_root(source, raw_root)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    practice = config["exercises"]["practice"]
    for entry in sorted(practice, key=lambda item: str(item["slug"])):
        slug = str(entry["slug"])
        exercise_dir = root / "exercises" / "practice" / slug
        instructions_path = exercise_dir / ".docs" / "instructions.md"
        example_path = exercise_dir / ".meta" / "example.py"
        if not instructions_path.is_file() or not example_path.is_file():
            yield ExercismRejection(slug, "missing_deterministic_pairing")
            continue
        try:
            prompt = normalize_python_bytes(
                instructions_path.read_bytes(),
                minimum_bytes=int(normalization["minimum_bytes"]),
                maximum_bytes=int(normalization["maximum_bytes"]),
                final_newline=bool(normalization["final_newline"]),
            )
            code = normalize_python_bytes(
                example_path.read_bytes(),
                minimum_bytes=int(normalization["minimum_bytes"]),
                maximum_bytes=int(normalization["maximum_bytes"]),
                final_newline=bool(normalization["final_newline"]),
            )
        except NormalizationError as error:
            yield ExercismRejection(slug, error.reason)
            continue
        combined = prompt + "\n" + code
        if scan_secrets(combined):
            yield ExercismRejection(slug, "secret_detected")
            continue
        if scan_pii(combined):
            yield ExercismRejection(slug, "pii_detected")
            continue
        if scan_unsafe_content(combined):
            yield ExercismRejection(slug, "unsafe_content")
            continue
        validation = validate_python(code)
        if not validation.ast_valid or not validation.tokenize_valid:
            yield ExercismRejection(slug, "invalid_python")
            continue
        quality = score_quality(code, validation, quality_config)
        if not quality.accepted:
            yield ExercismRejection(slug, quality.reason or "low_quality")
            continue
        practices = sorted(str(item) for item in entry.get("practices", []))
        category = ",".join(practices) if practices else "general"
        difficulty = DIFFICULTY_BUCKETS.get(int(entry.get("difficulty", 0)), "unspecified")
        messages = [
            Message("system", SYSTEM_MESSAGE),
            Message("user", prompt),
            Message("assistant", code),
        ]
        record = InstructionRecord(
            record_id=stable_record_id(source.id, slug, content_sha256(prompt + code)),
            messages=messages,
            category=category,
            difficulty=difficulty,
            tests=[],
            source_id=source.id,
            licence_spdx=source.dataset_level_licence,
            content_sha256=content_sha256(
                json.dumps(
                    [{"role": m.role, "content": m.content} for m in messages],
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
            generation_method="human",
            problem_family=slug,
        )
        record.validate()
        yield record


def exact_deduplicate(
    records: list[InstructionRecord],
) -> tuple[list[InstructionRecord], int]:
    """Keep the first record per content hash in slug order."""
    seen: dict[str, InstructionRecord] = {}
    duplicates = 0
    for record in records:
        if record.content_sha256 in seen:
            duplicates += 1
            continue
        seen[record.content_sha256] = record
    return list(seen.values()), duplicates


def near_deduplicate(
    records: list[InstructionRecord], *, threshold: float, shingle_size: int
) -> tuple[list[InstructionRecord], int, dict[str, list[str]]]:
    """O(n^2) pairwise near-duplicate filter; the Exercism corpus is small enough to compare
    directly rather than via the streaming LSH index used for large pretraining sources."""
    kept: list[InstructionRecord] = []
    kept_shingles: list[set[str]] = []
    clusters: dict[str, list[str]] = {}
    duplicates = 0
    for record in records:
        text = "\n".join(message.content for message in record.messages)
        shingles = token_shingles(text, shingle_size)
        removed = False
        for other, other_shingles in zip(kept, kept_shingles, strict=True):
            if jaccard_similarity(shingles, other_shingles) >= threshold:
                clusters.setdefault(other.record_id, []).append(record.record_id)
                duplicates += 1
                removed = True
                break
        if not removed:
            kept.append(record)
            kept_shingles.append(shingles)
    return kept, duplicates, clusters


def build_report(
    records: list[InstructionRecord],
    rejections: list[ExercismRejection],
    exact_duplicates: int,
    near_duplicates: int,
    near_clusters: dict[str, list[str]],
    leakage_passed: bool,
    split_counts: Counter[str],
) -> dict[str, Any]:
    """Summarize the Exercism instruction-adapter run."""
    category_distribution: Counter[str] = Counter(record.category for record in records)
    difficulty_distribution: Counter[str] = Counter(record.difficulty for record in records)
    rejection_reasons: Counter[str] = Counter(item.reason for item in rejections)
    return {
        "adapter": "exercism_instruction_pairing",
        "total_candidate_exercises": len(records) + len(rejections) + exact_duplicates,
        "paired_before_filters": len(records) + len(rejections),
        "rejections": dict(sorted(rejection_reasons.items())),
        "exact_duplicate_count": exact_duplicates,
        "near_duplicate_count": near_duplicates,
        "near_duplicate_clusters": near_clusters,
        "accepted_records": len(records),
        "instruction_category_distribution": dict(sorted(category_distribution.items())),
        "instruction_difficulty_distribution": dict(sorted(difficulty_distribution.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "leakage_passed": leakage_passed,
    }
