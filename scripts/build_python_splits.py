"""Build deterministic family-grouped train/validation/test JSONL files."""

from pathlib import Path
import argparse
import json
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genpy.data.io import load_examples, sha256_file, write_jsonl_atomic
from genpy.data.split import family_overlap, split_examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    examples = load_examples(args.input)
    splits = split_examples(examples, args.train_ratio, args.validation_ratio, args.test_ratio, args.seed)
    overlap = family_overlap(splits)
    if overlap:
        raise RuntimeError(f"Family leakage detected: {sorted(overlap)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("train", "validation", "test"):
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl_atomic(path, splits[name])
        hashes[name] = sha256_file(path)
        print(f"{name}: {len(splits[name])}")
    manifest = {
        "input": str(args.input), "seed": args.seed,
        "ratios": {"train": args.train_ratio, "validation": args.validation_ratio, "test": args.test_ratio},
        "counts": {name: len(records) for name, records in splits.items()},
        "family_leakage": sorted(overlap), "sha256": hashes,
    }
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
