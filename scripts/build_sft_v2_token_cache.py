"""Build v2 response-masked token caches; sanity is intentionally excluded."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genpy.data.io import sha256_file
from genpy.tokenizer import GenPyTokenizer
from scripts.build_sft_token_cache import build_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/instruction/sft_v2")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--output-dir", default="data/instruction/tokenized_v2")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--manifest", default="data/instruction/tokenized_v2/SFT_V2_TOKEN_CACHE_MANIFEST.json")
    args = parser.parse_args()
    analysis = json.loads((ROOT / "reports/checkpoint_8_v2/sequence_length_analysis.json").read_text(encoding="utf-8"))
    sequence_length = args.sequence_length or int(analysis["selected_sequence_length"])
    if sequence_length != int(analysis["selected_sequence_length"]):
        raise RuntimeError("v2 cache sequence length must match the recorded analysis")
    tokenizer_dir, dataset_dir, output_dir = ROOT / args.tokenizer, ROOT / args.dataset_dir, ROOT / args.output_dir
    tokenizer = GenPyTokenizer.load(tokenizer_dir)
    if tokenizer.vocab_size != 32000:
        raise RuntimeError("v2 cache requires GenPy-Tokenizer-32K")
    stats = {name: build_split(dataset_dir / f"{name}.jsonl", output_dir, tokenizer, sequence_length) for name in ("train", "validation", "challenge")}
    if any(value["truncation_count"] for value in stats.values()):
        raise RuntimeError("v2 cache contains truncation")
    manifest = {"format_version": 2, "dataset_version": "genpy-sft-v2-pilot-v1", "tokenizer_name": tokenizer.name, "tokenizer_vocab_size": tokenizer.vocab_size, "tokenizer_manifest_sha256": sha256_file(tokenizer_dir / "TOKENIZER_MANIFEST.json"), "sequence_length": sequence_length, "response_only_masking": True, "challenge_optimizer_use": False, "sanity_optimizer_use": False, "splits": stats}
    path = ROOT / args.manifest; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value[key] for key in ("document_count", "stored_token_count", "assistant_token_count", "ignored_prompt_token_count", "truncation_count")} for name, value in stats.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
