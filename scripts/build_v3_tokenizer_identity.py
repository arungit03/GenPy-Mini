"""Create platform-independent tokenizer identity and behavior fingerprints."""

from __future__ import annotations

import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v31_common import REPORT_DIR, TOKENIZER_DIR, behavior_vectors, frozen_source_hashes, sha256, tokenizer_identity, write_json_and_text


def main():
    manifest, identity, manifest_hash, semantic_hash = tokenizer_identity()
    vectors, behavior_hash = behavior_vectors()
    payload = {"format_version": 1, "tokenizer_name": manifest["tokenizer_name"], "tokenizer_version": manifest["tokenizer_version"], "tokenizer_type": manifest["tokenizer_type"], "vocab_size": manifest["vocab_size"], "special_token_ids": manifest["special_token_ids"], "artifact_sha256": manifest["artifact_sha256"], "dataset_train_sha256": manifest["dataset_train_sha256"], "tokenizer_manifest_canonical_sha256": manifest_hash, "tokenizer_semantic_identity": identity, "tokenizer_semantic_identity_sha256": semantic_hash, "tokenization_behavior": vectors, "tokenization_behavior_sha256": behavior_hash, "raw_manifest_sha256_observational": sha256(TOKENIZER_DIR / "TOKENIZER_MANIFEST.json"), "source_hashes_at_start": frozen_source_hashes()}
    write_json_and_text(REPORT_DIR / "tokenizer_identity.json", REPORT_DIR / "tokenizer_identity.txt", "GenPy Checkpoint 8-v3.1 tokenizer identity", payload)
    print(json.dumps({key: payload[key] for key in ("tokenizer_name", "tokenizer_version", "vocab_size", "tokenizer_manifest_canonical_sha256", "tokenizer_semantic_identity_sha256", "tokenization_behavior_sha256")}, indent=2))


if __name__ == "__main__": main()
