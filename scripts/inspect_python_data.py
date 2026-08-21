"""Display a deterministic, human-readable sample from a JSONL dataset."""

from pathlib import Path
import argparse
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genpy.data.io import load_examples
from genpy.data.validate import apply_validation, code_for


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    examples = load_examples(args.file)
    selected = random.Random(args.seed).sample(examples, min(args.samples, len(examples)))
    for number, example in enumerate(selected, 1):
        result = apply_validation(example)
        print(f"Example {number}\n---------")
        print(f"ID: {example.id}")
        print(f"Category: {example.category}")
        print(f"Task: {example.task_type}")
        print(f"Family: {example.family_id}")
        print(f"Instruction: {getattr(example, 'instruction', '[code-only example]')}")
        print(f"Response:\n{code_for(example)}")
        print(f"Source: {example.source}")
        print(f"Quality: {example.quality_score}")
        print(f"Syntax: {'PASS' if result.syntax_valid else 'FAIL'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
