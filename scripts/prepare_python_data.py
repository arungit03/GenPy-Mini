"""Prepare normalized, validated, deduplicated Python JSONL data."""

from pathlib import Path
import argparse
import json
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genpy.data.deduplicate import deduplicate_examples
from genpy.data.io import iter_records, write_jsonl_atomic
from genpy.data.normalize import normalize_example
from genpy.data.registry import ExclusionRegistry
from genpy.data.schema import example_from_mapping
from genpy.data.statistics import classify_dataset_size, compute_statistics
from genpy.data.validate import apply_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strict-code", action="store_true")
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42, help="Reserved for deterministic pipeline versioning")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exclusions", type=Path, default=None)
    args = parser.parse_args()
    del args.seed  # The current transformations are order-preserving and deterministic.
    raw_count = 0
    rejected = 0
    syntax_failures = 0
    non_python = 0
    examples = []
    exclusion_registry = ExclusionRegistry.from_json(args.exclusions) if args.exclusions else ExclusionRegistry()
    for raw in iter_records(args.input):
        raw_count += 1
        try:
            example = normalize_example(example_from_mapping(raw))
            if exclusion_registry.is_excluded(example):
                rejected += 1
                continue
            result = apply_validation(example, strict_code=args.strict_code)
            syntax_failures += not result.syntax_valid
            non_python += result.non_python_suspected
            if not result.valid:
                rejected += 1
                continue
            examples.append(example)
        except (TypeError, ValueError, KeyError) as exc:
            rejected += 1
            print(f"Rejected record {raw_count}: {exc}", file=sys.stderr)
    dedup_report = None
    if args.deduplicate:
        examples, dedup_report = deduplicate_examples(
            examples, near_duplicate=True, near_duplicate_threshold=args.near_duplicate_threshold
        )
    output_hash = write_jsonl_atomic(args.output, examples)
    stats = compute_statistics(examples, examples_rejected=rejected)
    report = {
        "total_raw": raw_count,
        "total_valid_before_deduplication": raw_count - rejected,
        "total_rejected": rejected,
        "syntax_failures": int(syntax_failures),
        "non_python_suspected": int(non_python),
        "final_examples": len(examples),
        "dataset_classification": classify_dataset_size(len(examples)),
        "output_sha256": output_hash,
        "statistics": stats.to_dict(),
    }
    if dedup_report:
        report.update({
            "exact_duplicates": dedup_report.exact_duplicates,
            "instruction_duplicates": dedup_report.instruction_duplicates,
            "code_duplicates": dedup_report.code_duplicates,
            "near_duplicates": dedup_report.near_duplicates,
            "duplicates_removed": dedup_report.examples_removed,
        })
    if args.report:
        report_path = args.output.with_suffix(args.output.suffix + ".report.json")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Raw examples: {raw_count}")
    print(f"Accepted before deduplication: {raw_count - rejected}")
    print(f"Rejected: {rejected}")
    print(f"Final examples: {len(examples)}")
    print(f"Output SHA-256: {output_hash}")
    print(f"Dataset classification: {classify_dataset_size(len(examples))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
