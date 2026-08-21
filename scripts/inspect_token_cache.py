"""Inspect cache statistics, hashes, ranges, and a few decoded windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.data.io import sha256_file
from genpy.tokenizer import GenPyTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", default="data/tokenized/genpy-32k/TOKEN_CACHE_MANIFEST.json"); args = parser.parse_args()
    manifest_path = ROOT / args.manifest; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); root = manifest_path.parent
    tokenizer = GenPyTokenizer.load(ROOT / "artifacts/tokenizer/genpy-32k")
    for split in ("train", "validation"):
        path = root / f"{split}.bin"; tokens = np.memmap(path, dtype=np.uint16, mode="r")
        print(f"{split}: {len(tokens):,} tokens; range {int(tokens.min())}-{int(tokens.max())}; hash {'PASS' if sha256_file(path) == manifest[f'{split}_bin_sha256'] else 'FAIL'}")
        print(f"  BOS={int(np.count_nonzero(tokens == tokenizer.bos_token_id)):,}; EOS={int(np.count_nonzero(tokens == tokenizer.eos_token_id)):,}; sample={tokenizer.decode(tokens[:min(32, len(tokens))].astype(int).tolist())!r}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
