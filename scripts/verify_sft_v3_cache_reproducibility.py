"""Rebuild the v3.1 train/validation cache twice in temporary directories."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genpy.tokenizer import GenPyTokenizer
from scripts.build_sft_token_cache import build_split
from scripts.v31_common import TOKENIZER_DIR, V3_SFT, sha256


def main():
    sequence = json.loads((ROOT / "reports/checkpoint_8_v3_1/sequence_length_analysis.json").read_text())["selected_sequence_length"]
    tokenizer = GenPyTokenizer.load(TOKENIZER_DIR)
    with tempfile.TemporaryDirectory(prefix="genpy-v31-cache-a-") as first, tempfile.TemporaryDirectory(prefix="genpy-v31-cache-b-") as second:
        first_dir, second_dir = Path(first), Path(second)
        for output in (first_dir, second_dir):
            for name in ("train", "validation"): build_split(V3_SFT / f"{name}.jsonl", output, tokenizer, sequence)
        files = [f"{split}.{suffix}" for split in ("train", "validation") for suffix in ("input_ids.bin", "labels.bin", "offsets.npy")]
        hashes = {name: {"first": sha256(first_dir / name), "second": sha256(second_dir / name), "equal": sha256(first_dir / name) == sha256(second_dir / name)} for name in files}
    result = {"cache_reproducibility_pass": all(value["equal"] for value in hashes.values()), "files": hashes, "temporary_caches_removed": True}
    path = ROOT / "reports/checkpoint_8_v3_1/cache_reproducibility.json"; path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (ROOT / "reports/checkpoint_8_v3_1/cache_reproducibility.txt").write_text("GenPy Checkpoint 8-v3.1 cache reproducibility\n\n" + json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["cache_reproducibility_pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
