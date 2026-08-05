from __future__ import annotations

import json
from pathlib import Path

from genpy.data.exercism_adapter import (
    ExercismRejection,
    exact_deduplicate,
    iter_exercise_pairs,
    near_deduplicate,
)
from genpy.data.schemas import InstructionRecord
from genpy.data.source_registry import SourceEntry

_NORMALIZATION = {"minimum_bytes": 4, "maximum_bytes": 100000, "final_newline": True}
_QUALITY = {
    "minimum_score": 0.1,
    "maximum_line_length": 300,
    "maximum_long_line_ratio": 0.5,
    "maximum_repeated_line_ratio": 0.9,
    "minimum_lexical_tokens": 1,
    "generated_markers": ["do not edit"],
}


def _source() -> SourceEntry:
    return SourceEntry(
        id="exercism_test_fixture",
        name="Exercism test fixture",
        official_url="https://example.invalid/exercism-fixture",
        dataset_card_url=None,
        version="fixture-v1",
        access_method="github_archive",
        archive_url="https://example.invalid/exercism-fixture/zip/fixture-v1",
        checksum_sha256="unknown",
        expected_download_size="unknown",
        expected_extracted_size="unknown",
        expected_usable_records="unknown",
        estimated_disk_requirement="unknown",
        streaming_supported=True,
        languages=("Python",),
        dataset_level_licence="MIT",
        per_record_licence=True,
        provenance_available=True,
        opt_out_supported=False,
        attribution_required=True,
        status="approved_instruction_only",
        review_notes="Fixture-only; never production data.",
        repository="exercism/fixture",
    )


def _write_exercise(root: Path, slug: str, instructions: str, example: str) -> None:
    exercise_dir = root / "exercises" / "practice" / slug
    (exercise_dir / ".docs").mkdir(parents=True)
    (exercise_dir / ".meta").mkdir(parents=True)
    (exercise_dir / ".docs" / "instructions.md").write_text(instructions, encoding="utf-8")
    (exercise_dir / ".meta" / "example.py").write_text(example, encoding="utf-8")


def _build_fixture_tree(raw_root: Path, source: SourceEntry) -> None:
    root = raw_root / source.id / source.version / "fixture-repo"
    _write_exercise(
        root,
        "double",
        "# Instructions\n\nWrite a function that doubles a number.\n",
        "def double(number):\n    return number * 2\n",
    )
    _write_exercise(
        root,
        "triple",
        "# Instructions\n\nWrite a function that triples a number.\n",
        "def triple(number):\n    return number * 3\n",
    )
    _write_exercise(
        root,
        "leaky-secret",
        "# Instructions\n\nParse a config file.\n",
        'password = "hunter22345"\n',
    )
    config = {
        "exercises": {
            "practice": [
                {"slug": "double", "difficulty": 1, "practices": ["numbers"]},
                {"slug": "triple", "difficulty": 4, "practices": []},
                {"slug": "leaky-secret", "difficulty": 2, "practices": []},
            ]
        }
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_iter_exercise_pairs_builds_valid_records_and_rejects_secrets(tmp_path: Path) -> None:
    source = _source()
    _build_fixture_tree(tmp_path, source)

    records: list[InstructionRecord] = []
    rejections: list[ExercismRejection] = []
    for item in iter_exercise_pairs(source, tmp_path, _NORMALIZATION, _QUALITY):
        if isinstance(item, ExercismRejection):
            rejections.append(item)
        else:
            item.validate()
            records.append(item)

    assert {record.problem_family for record in records} == {"double", "triple"}
    assert [r.slug for r in rejections] == ["leaky-secret"]
    assert rejections[0].reason == "secret_detected"
    double = next(r for r in records if r.problem_family == "double")
    assert double.difficulty == "beginner"
    triple = next(r for r in records if r.problem_family == "triple")
    assert triple.difficulty == "intermediate"


def test_exact_and_near_deduplicate_collapse_identical_pairs(tmp_path: Path) -> None:
    source = _source()
    root = tmp_path / source.id / source.version / "fixture-repo"
    _write_exercise(
        root, "a", "# Instructions\n\nAdd one.\n", "def add_one(n):\n    return n + 1\n"
    )
    _write_exercise(
        root, "b", "# Instructions\n\nAdd one.\n", "def add_one(n):\n    return n + 1\n"
    )
    config = {
        "exercises": {
            "practice": [
                {"slug": "a", "difficulty": 1, "practices": []},
                {"slug": "b", "difficulty": 1, "practices": []},
            ]
        }
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    records = [
        item
        for item in iter_exercise_pairs(source, tmp_path, _NORMALIZATION, _QUALITY)
        if isinstance(item, InstructionRecord)
    ]
    assert len(records) == 2

    deduplicated, near_count, _clusters = near_deduplicate(
        records, threshold=0.85, shingle_size=3
    )
    assert near_count == 1
    assert len(deduplicated) == 1

    exact_result, exact_count = exact_deduplicate(records)
    assert exact_count == 1
    assert len(exact_result) == 1
