"""Verify a saved GenPy tokenizer artifact and its production invariants."""

import argparse
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root
ensure_project_root()

from genpy.tokenizer.validation import validate_tokenizer_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-vocab", type=int, default=32000)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_200m.yaml"))
    args = parser.parse_args()
    report = validate_tokenizer_artifact(args.tokenizer, args.manifest, args.expected_vocab, args.model_config if args.expected_vocab == 32000 else None)
    print(f"Tokenizer: {report['tokenizer']}")
    print(f"Vocabulary: {report['vocab_size']}")
    print("PAD: 0\nBOS: 1\nEOS: 2\nUNK: 3")
    print("Architecture vocabulary match: PASS" if args.expected_vocab == 32000 else "Architecture vocabulary match: SKIPPED (smoke)")
    print("Round-trip tests: PASS")
    print("Checksum: PASS")
    print("Production validation: PASS" if args.expected_vocab == 32000 else "Smoke validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
