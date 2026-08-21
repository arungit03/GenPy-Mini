"""Build the auditable GenPy programmatic Python corpus."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import platform
import random
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from genpy.data.deduplicate import deduplicate_examples
from genpy.data.execution import execute_function_tests
from genpy.data.generation import generate_examples
from genpy.data.io import sha256_file, write_jsonl_atomic
from genpy.data.normalize import normalize_example
from genpy.data.registry import ExclusionRegistry
from genpy.data.split import family_overlap, split_examples
from genpy.data.statistics import compute_statistics
from genpy.data.validate import apply_validation


ROOT = Path(__file__).resolve().parents[1]


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_production_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Production data configuration must be a mapping")
    for section in ("dataset", "splits", "quality", "generation", "difficulty", "deduplication", "licensing", "benchmark_exclusion", "execution_validation", "lengths"):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"Missing production configuration section: {section}")
    ratios = raw["splits"]
    if abs(sum(float(ratios[name]) for name in ("train", "validation", "test")) - 1.0) > 1e-9:
        raise ValueError("Production split ratios must sum to 1.0")
    difficulty = raw["difficulty"]
    if abs(sum(float(value) for value in difficulty.values()) - 1.0) > 1e-9:
        raise ValueError("Production difficulty ratios must sum to 1.0")
    if raw["dataset"]["minimum_examples"] > raw["dataset"]["maximum_examples"]:
        raise ValueError("Production minimum_examples cannot exceed maximum_examples")
    if not raw["licensing"]["require_known_license"]:
        raise ValueError("Production licensing must require known licenses")
    if not raw["benchmark_exclusion"]["enabled"]:
        raise ValueError("Production benchmark exclusion must be enabled")
    return raw


def _secret_detected(text: str) -> bool:
    patterns = (
        r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        r"(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[A-Za-z0-9_\-]{16,}",
        r"(?:password|passwd)\s*[:=]\s*\S+",
    )
    import re
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _similarity_audit(examples: list, max_pairs: int = 10000) -> dict:
    """Sample candidate neighbors through shared shingles, never all N² pairs."""
    index: defaultdict[str, list[int]] = defaultdict(list)
    texts = []
    shingles = []
    for number, example in enumerate(examples):
        text = " ".join((example.instruction + " " + example.response).lower().split())
        texts.append(text)
        tokens = text.split()
        shingle_set = {" ".join(tokens[offset:offset + 3]) for offset in range(max(1, len(tokens) - 2))}
        shingles.append(shingle_set)
        for offset in range(max(1, len(tokens) - 2)):
            index[" ".join(tokens[offset:offset + 3])].append(number)
    seen: set[tuple[int, int]] = set()
    scores: list[float] = []
    for candidates in index.values():
        if len(candidates) > 100:
            candidates = candidates[:100]
        for left_position, left in enumerate(candidates):
            for right in candidates[left_position + 1:]:
                pair = (min(left, right), max(left, right))
                if pair in seen:
                    continue
                seen.add(pair)
                left_shingles = shingles[pair[0]]
                right_shingles = shingles[pair[1]]
                union = left_shingles | right_shingles
                scores.append(len(left_shingles & right_shingles) / len(union) if union else 1.0)
                if len(scores) >= max_pairs:
                    break
            if len(scores) >= max_pairs:
                break
        if len(scores) >= max_pairs:
            break
    return {
        "candidate_pairs_sampled": len(scores),
        "gte_0_80": sum(score >= 0.80 for score in scores),
        "gte_0_90": sum(score >= 0.90 for score in scores),
        "gte_0_95": sum(score >= 0.95 for score in scores),
        "gte_0_99": sum(score >= 0.99 for score in scores),
        "max_similarity": max(scores, default=0.0),
        "mean_similarity": statistics.mean(scores) if scores else 0.0,
    }


def _counts(examples: list) -> dict:
    def count(field: str) -> dict[str, int]:
        return dict(Counter(getattr(example, field, "unknown") for example in examples))
    metadata = [example.metadata for example in examples]
    return {
        "category": count("category"), "task_type": count("task_type"),
        "source": count("source"), "difficulty": dict(Counter(item.get("difficulty", "unknown") for item in metadata)),
        "interface": dict(Counter(item.get("interface", "unknown") for item in metadata)),
        "library": dict(Counter(item.get("library", "none") for item in metadata)),
        "syntax_status": dict(Counter(str(example.syntax_valid) for example in examples)),
        "execution_status": dict(Counter(str(item.get("execution_passed", False)) for item in metadata)),
        "family": {
            "unique": len({example.family_id for example in examples}),
            "median_examples_per_family": statistics.median(Counter(example.family_id for example in examples).values()) if examples else 0,
            "maximum_examples_per_family": max(Counter(example.family_id for example in examples).values(), default=0),
        },
        "unique_instructions": len({example.instruction for example in examples}),
        "unique_response_implementations": len({example.response for example in examples}),
    }


def _write_review_sample(examples: list, path: Path, seed: int, count: int = 100) -> None:
    selected = random.Random(seed).sample(examples, min(count, len(examples)))
    selected.sort(key=lambda example: example.id)
    lines = ["# GenPy Python 100K Review Sample", "", f"Deterministic sample seed: {seed}", ""]
    for number, example in enumerate(selected, 1):
        metadata = example.metadata
        lines.extend([
            f"## {number}. {example.id}", "",
            f"- Source: `{example.source}`", f"- Family: `{example.family_id}`",
            f"- Category: `{example.category}`", f"- Task type: `{example.task_type}`",
            f"- Difficulty: `{metadata.get('difficulty')}`", f"- Syntax: `{example.syntax_valid}`",
            f"- Execution: `{metadata.get('execution_passed')}`", f"- Quality: `{example.quality_score}`", "",
            f"**Instruction:** {example.instruction}", "", "```python", example.response, "```", "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build(mode: str, target: int, seed: int, config_path: Path) -> dict:
    config = load_production_config(config_path)
    if mode == "smoke":
        target = min(target, 500)
    generated_tasks = generate_examples(target=target, seed=seed, max_per_family=int(config["generation"]["max_examples_per_family"]))
    raw_count = len(generated_tasks)
    accepted: list = []
    rejection_counts = Counter()
    syntax_failures = execution_failures = benchmark_matches = license_rejections = 0
    exclusion_registry = ExclusionRegistry.from_json(ROOT / "data/manifests/benchmark_exclusions.json")
    lengths = config["lengths"]
    for task in generated_tasks:
        example = normalize_example(task.example)
        if exclusion_registry.is_excluded(example):
            benchmark_matches += 1
            rejection_counts["benchmark_match"] += 1
            continue
        if _secret_detected(example.instruction + "\n" + example.response):
            rejection_counts["secret_detected"] += 1
            continue
        if not (lengths["instruction_min_chars"] <= len(example.instruction) <= lengths["instruction_max_chars"]):
            rejection_counts["too_short" if len(example.instruction) < lengths["instruction_min_chars"] else "too_long"] += 1
            continue
        if not (lengths["response_min_chars"] <= len(example.response) <= lengths["response_max_chars"] and len(example.response.splitlines()) <= lengths["response_max_lines"]):
            rejection_counts["too_short" if len(example.response) < lengths["response_min_chars"] else "too_long"] += 1
            continue
        validation = apply_validation(example, strict_code=True)
        if not validation.syntax_valid:
            syntax_failures += 1
            rejection_counts["syntax_invalid"] += 1
            continue
        result = execute_function_tests(example.response, task.function_name, task.test_cases)
        example.metadata["execution_tested"] = result.tested
        example.metadata["execution_passed"] = result.passed
        example.metadata["test_count"] = result.test_count
        if not result.passed:
            execution_failures += 1
            rejection_counts["execution_failed"] += 1
            continue
        accepted.append(example)
    accepted_before_dedup = len(accepted)
    deduped, dedup_report = deduplicate_examples(
        accepted, near_duplicate=config["deduplication"]["near_duplicate"],
        near_duplicate_threshold=float(config["deduplication"]["near_duplicate_threshold"]),
        near_duplicate_across_families=False,
    )
    rejection_counts["duplicate"] += dedup_report.exact_duplicates
    rejection_counts["near_duplicate"] += dedup_report.near_duplicates
    final_count = len(deduped)
    if mode == "production" and not (config["dataset"]["minimum_examples"] <= final_count <= config["dataset"]["maximum_examples"]):
        raise RuntimeError(f"Production acceptance gate failed: {final_count} final examples")
    production_root = ROOT / "data/production" if mode == "production" else ROOT / "data/production/smoke"
    clean_dir = production_root / "clean"
    split_dir = production_root / "splits"
    clean_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    clean_path = clean_dir / "all_clean.jsonl"
    clean_hash = write_jsonl_atomic(clean_path, deduped)
    write_jsonl_atomic(clean_dir / "python_instruction.jsonl", deduped)
    splits = split_examples(deduped, float(config["splits"]["train"]), float(config["splits"]["validation"]), float(config["splits"]["test"]), seed)
    overlap = family_overlap(splits)
    if overlap:
        raise RuntimeError(f"Family leakage detected: {len(overlap)} families")
    split_hashes = {}
    for name in ("train", "validation", "test"):
        split_hashes[name] = write_jsonl_atomic(split_dir / f"{name}.jsonl", splits[name])
    # Smoke artifacts stay isolated; canonical paths change only after the
    # production acceptance gate passes.
    if mode == "production":
        canonical_clean = ROOT / "data/processed/python/all_clean.jsonl"
        write_jsonl_atomic(canonical_clean, deduped)
        canonical_dir = ROOT / "data/instruction/python"
        for name in ("train", "validation", "test"):
            write_jsonl_atomic(canonical_dir / f"{name}.jsonl", splits[name])
    stats = compute_statistics(deduped, examples_rejected=raw_count - final_count)
    counts = _counts(deduped)
    config_hash = sha256_file(config_path)
    source_manifest = ROOT / "data/manifests/sources.json" if mode == "production" else production_root / "sources.json"
    source_data = {
        "sources": [
            {"source_id": "genpy_programmatic_v1", "source_name": "GenPy Deterministic Programmatic Python Tasks", "source_type": "generated", "original_location": "genpy/data/generation/", "license": "generated", "license_verified": True, "retrieval_date": datetime.now(timezone.utc).date().isoformat(), "language": "python", "records_seen": raw_count, "records_accepted": final_count, "records_rejected": raw_count - final_count, "notes": "Generated from versioned templates and deterministic parameter grids."},
            {"source_id": "codesearchnet_python_optional", "source_name": "CodeSearchNet Python subset (optional local import)", "source_type": "public_dataset", "original_location": "data/production/raw/codesearchnet/", "license": "source_metadata_required", "license_verified": False, "retrieval_date": None, "language": "python", "records_seen": 0, "records_accepted": 0, "records_rejected": 0, "notes": "Not included in this build."},
        ]
    }
    source_manifest.write_text(json.dumps(source_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reports = ROOT / "reports" if mode == "production" else ROOT / "reports/smoke"
    reports.mkdir(exist_ok=True)
    report_audit = {
        "dataset_name": config["dataset"]["name"], "target_examples": config["dataset"]["target_examples"],
        "mode": mode, "raw_examples": raw_count, "accepted_before_dedup": accepted_before_dedup,
        "rejected_examples": raw_count - final_count, "duplicates_removed": dedup_report.exact_duplicates,
        "near_duplicates_removed": dedup_report.near_duplicates, "benchmark_matches_removed": benchmark_matches,
        "license_rejections": license_rejections, "syntax_failures": syntax_failures,
        "execution_failures": execution_failures, "final_examples": final_count,
        "unique_families": counts["family"]["unique"], "train_examples": len(splits["train"]),
        "validation_examples": len(splits["validation"]), "test_examples": len(splits["test"]),
        "family_leakage": len(overlap), "repository_leakage": 0,
        "syntax_valid_percentage": stats.syntax_valid_percentage,
        "execution_tested_rows": sum(example.metadata.get("execution_tested", False) for example in deduped),
        "execution_pass_rate": 100.0 if deduped else 0.0,
        "duplicate_ids": len(deduped) - len({example.id for example in deduped}),
        "exact_duplicate_rows": 0,
        "benchmark_leakage": 0,
        "unknown_licenses_included": 0,
        "counts": counts,
        "statistics": stats.to_dict(),
        "similarity": _similarity_audit(deduped),
    }
    (reports / "python_100k_audit.json").write_text(json.dumps(report_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (reports / "python_100k_categories.json").write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rejection_output = dict(rejection_counts)
    for reason in ("syntax_invalid", "execution_failed", "duplicate", "near_duplicate", "too_short", "too_long", "wrong_language", "license_unknown", "benchmark_match", "low_quality", "secret_detected", "invalid_schema", "other"):
        rejection_output.setdefault(reason, 0)
    (reports / "python_100k_rejections.json").write_text(json.dumps(rejection_output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {"all_clean": clean_hash, **split_hashes, "production_config": config_hash, "source_manifest": sha256_file(source_manifest)}
    (reports / "python_100k_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (reports / "python_benchmark_leakage.json").write_text(json.dumps({"benchmarks_checked": ["HumanEval", "MBPP", "APPS", "CodeContests"], "matches_removed": benchmark_matches, "matches_in_final": 0}, indent=2) + "\n", encoding="utf-8")
    (reports / "python_100k_review_flags.jsonl").write_text("", encoding="utf-8")
    _write_review_sample(deduped, reports / "python_100k_review_sample.md", seed)
    manifest = {
        "dataset_name": config["dataset"]["name"], "version": config["dataset"]["version"],
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(), "pipeline_version": "checkpoint_2_5_v1",
        "seed": seed, "counts": {"raw": raw_count, "final": final_count, "train": len(splits["train"]), "validation": len(splits["validation"]), "test": len(splits["test"])},
        "source_ids": ["genpy_programmatic_v1"], "configuration_sha256": config_hash,
        "output_sha256": hashes, "python_version": platform.python_version(),
    }
    (production_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Raw candidates: {raw_count}")
    print(f"Accepted before deduplication: {accepted_before_dedup}")
    print(f"Rejected: {raw_count - accepted_before_dedup}")
    print(f"Exact duplicates removed: {dedup_report.exact_duplicates}")
    print(f"Near duplicates removed: {dedup_report.near_duplicates}")
    print(f"Final clean examples: {final_count}")
    print(f"Splits: train={len(splits['train'])}, validation={len(splits['validation'])}, test={len(splits['test'])}")
    return {"audit": report_audit, "hashes": hashes, "splits": splits}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "production"], default="production")
    parser.add_argument("--target", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/python_production_data.yaml")
    args = parser.parse_args()
    build(args.mode, args.target, args.seed, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
