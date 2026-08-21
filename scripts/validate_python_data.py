"""Validate a prepared Python JSONL file and return CI-friendly status."""

from pathlib import Path
import argparse
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genpy.data.io import iter_records
from genpy.data.normalize import normalize_example
from genpy.data.registry import CATEGORIES
from genpy.data.schema import example_from_mapping
from genpy.data.validate import apply_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--strict-code", action="store_true")
    args = parser.parse_args()
    total = valid = syntax_failures = missing_fields = duplicate_ids = invalid_categories = non_python = 0
    ids: set[str] = set()
    for raw in iter_records(args.file):
        total += 1
        try:
            example = normalize_example(example_from_mapping(raw))
            if example.id in ids:
                duplicate_ids += 1
            ids.add(example.id)
            if not getattr(example, "instruction", getattr(example, "code", "")):
                missing_fields += 1
            if example.category not in CATEGORIES:
                invalid_categories += 1
            result = apply_validation(example, strict_code=args.strict_code)
            syntax_failures += not result.syntax_valid
            non_python += result.non_python_suspected
            valid += result.valid
        except (TypeError, ValueError, KeyError):
            missing_fields += 1
    invalid = total - valid
    print(f"Total examples: {total}")
    print(f"Valid examples: {valid}")
    print(f"Invalid examples: {invalid}")
    print(f"Syntax failures: {syntax_failures}")
    print(f"Missing fields: {missing_fields}")
    print(f"Duplicate IDs: {duplicate_ids}")
    print(f"Invalid categories: {invalid_categories}")
    print(f"Non-Python suspected: {non_python}")
    passed = invalid == 0 and duplicate_ids == 0 and invalid_categories == 0
    print(f"Validation result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
