"""Validate a GenPy tokenizer artifact against its production contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.data.io import iter_records
from genpy.tokenizer.config import load_tokenizer_config
from genpy.tokenizer.corpus import format_document
from genpy.tokenizer.tokenizer import GenPyTokenizer
from genpy.tokenizer.validation import measure_texts


ROUNDTRIP_SAMPLES = [
    "hello, world!",
    "def greet(name):\n    if name:\n        print(\"Hello\", name)\n    else:\n        print(\"Hello\")",
    "\tvalue = [1, 2, 3]\n\n# café 🙂\n",
    "__name__ == \"__main__\"\nx //= 2\na ** b\nitems[::-1]",
    "print(\"hello\\nworld\")\npath = \"C:\\\\Users\\\\name\"\ntext = \"\"\"multiple\nlines\nhere\"\"\"",
]


def split_metrics(tokenizer: GenPyTokenizer, path: Path, config) -> dict:
    texts = [format_document(record, config) for record in iter_records(path)]
    return measure_texts(tokenizer, texts).to_dict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--train", default="data/instruction/python/train.jsonl")
    parser.add_argument("--validation", default="data/instruction/python/validation.jsonl")
    parser.add_argument("--test", default="data/instruction/python/test.jsonl")
    parser.add_argument("--config", default="configs/tokenizer.yaml")
    args = parser.parse_args()
    tokenizer = GenPyTokenizer.load(ROOT / args.tokenizer)
    config = load_tokenizer_config(ROOT / args.config)
    roundtrip = all(tokenizer.decode(tokenizer.encode(sample)) == sample for sample in ROUNDTRIP_SAMPLES)
    metrics = {name: split_metrics(tokenizer, ROOT / path, config) for name, path in {
        "train": args.train, "validation": args.validation, "test": args.test}.items()}
    reload_tokenizer = GenPyTokenizer.load(ROOT / args.tokenizer)
    stability = all(tokenizer.encode(sample) == reload_tokenizer.encode(sample) for sample in ROUNDTRIP_SAMPLES)
    result = {
        "tokenizer": tokenizer.name, "type": "Byte-Level BPE", "vocabulary_size": tokenizer.vocab_size,
        "special_token_ids": {"PAD": tokenizer.pad_token_id, "BOS": tokenizer.bos_token_id, "EOS": tokenizer.eos_token_id, "UNK": tokenizer.unk_token_id},
        "roundtrip_pass": roundtrip, "save_reload_stability": stability, "metrics": metrics,
        "vocabulary_size_pass": tokenizer.vocab_size == 32000,
        "special_token_ids_pass": tokenizer.pad_token_id == 0 and tokenizer.bos_token_id == 1 and tokenizer.eos_token_id == 2 and tokenizer.unk_token_id == 3,
    }
    report_path = ROOT / "reports/tokenizer_validation.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("GenPy Tokenizer Validation\n==========================")
    print(f"Tokenizer: {tokenizer.name}\nType: Byte-Level BPE\nVocabulary: {tokenizer.vocab_size}")
    print("Special Tokens:\nPAD = 0\nBOS = 1\nEOS = 2\nUNK = 3")
    print(f"Round-trip tests: {'PASS' if roundtrip else 'FAIL'}")
    for name in ("train", "validation", "test"):
        metric = metrics[name]
        print(f"{name.title()} unknown-token rate: {metric['unknown_rate']:.6%}")
    checks = [roundtrip, stability, result["vocabulary_size_pass"], result["special_token_ids_pass"]] + [m["unknown_rate"] == 0.0 for m in metrics.values()]
    passed = all(checks)
    print(f"Vocabulary size: {'PASS' if result['vocabulary_size_pass'] else 'FAIL'}")
    print(f"Special token IDs: {'PASS' if result['special_token_ids_pass'] else 'FAIL'}")
    print(f"Final result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
