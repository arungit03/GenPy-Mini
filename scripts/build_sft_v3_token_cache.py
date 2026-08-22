"""Build and hard-validate the train/validation-only v3.1 SFT cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genpy.tokenizer import GenPyTokenizer
from scripts.build_sft_token_cache import build_split
from scripts.v31_common import REPORT_DIR, TOKENIZER_DIR, V3_SFT, canonical_sha256, sha256, tokenizer_identity, behavior_vectors, write_json_and_text


def validate_cache(cache_dir, name, metadata, sequence_length):
    input_path, labels_path, offsets_path = (cache_dir / f"{name}.{suffix}" for suffix in ("input_ids.bin", "labels.bin", "offsets.npy"))
    inputs = np.memmap(input_path, mode="r", dtype=np.uint16); labels = np.memmap(labels_path, mode="r", dtype=np.int32); offsets = np.load(offsets_path, mmap_mode="r")
    if len(offsets) != metadata["document_count"] + 1 or int(offsets[0]) != 0 or int(offsets[-1]) != len(inputs) or int(offsets[-1]) != len(labels) or np.any(np.diff(offsets) < 0): raise RuntimeError(f"{name}: invalid offsets")
    if len(inputs) and (int(inputs.min()) < 0 or int(inputs.max()) >= 32000): raise RuntimeError(f"{name}: input ID out of range")
    if len(labels) and (int(labels[labels != -100].min()) < 0 or int(labels[labels != -100].max()) >= 32000): raise RuntimeError(f"{name}: label out of range")
    document_lengths = np.diff(offsets)
    if any(length <= 1 or length - 1 > sequence_length for length in document_lengths): raise RuntimeError(f"{name}: stored sequence exceeds selected length or is empty")
    for start, end in zip(offsets[:-1], offsets[1:]):
        document_labels = labels[int(start):int(end)]
        if not np.any(document_labels == -100) or not np.any(document_labels >= 0): raise RuntimeError(f"{name}: response-only mask missing prompt or assistant labels")
    metadata.update({"input_ids_sha256": sha256(input_path), "labels_sha256": sha256(labels_path), "offsets_sha256": sha256(offsets_path), "offsets_valid": True, "input_id_min": int(inputs.min()) if len(inputs) else None, "input_id_max": int(inputs.max()) if len(inputs) else None, "label_min": int(labels[labels != -100].min()) if np.any(labels != -100) else None, "label_max": int(labels[labels != -100].max()) if np.any(labels != -100) else None, "maximum_stored_input_positions": int(document_lengths.max() - 1), "maximum_stored_sequence": int(document_lengths.max()), "all_documents_have_prompt_and_assistant_labels": True})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="data/instruction/tokenized_v3"); parser.add_argument("--manifest", default="data/instruction/tokenized_v3/SFT_V3_TOKEN_CACHE_MANIFEST.json"); args = parser.parse_args()
    output = ROOT / args.output_dir; output.mkdir(parents=True, exist_ok=True)
    forbidden = [output / f"{name}.{suffix}" for name in ("challenge", "sanity") for suffix in ("input_ids.bin", "labels.bin", "offsets.npy")]
    if any(path.exists() for path in forbidden): raise RuntimeError("challenge/sanity cache artifacts already exist; refusing to continue")
    analysis = json.loads((REPORT_DIR / "sequence_length_analysis.json").read_text(encoding="utf-8")); sequence_length = int(analysis["selected_sequence_length"])
    tokenizer = GenPyTokenizer.load(TOKENIZER_DIR)
    manifest_data, identity, canonical_manifest, semantic_identity = tokenizer_identity(); vectors, behavior_hash = behavior_vectors()
    stats = {name: build_split(V3_SFT / f"{name}.jsonl", output, tokenizer, sequence_length) for name in ("train", "validation")}
    for name in stats: validate_cache(output, name, stats[name], sequence_length)
    for path in forbidden:
        if path.exists(): raise RuntimeError(f"forbidden evaluation cache created: {path}")
    manifest = {"format_version": 3, "dataset_name": "GenPy-SFT-v3-Semantic", "dataset_version": "genpy-sft-v3-semantic-v1", "tokenizer_name": identity["tokenizer_name"], "tokenizer_version": identity["tokenizer_version"], "tokenizer_vocab_size": identity["vocab_size"], "bos_token_id": identity["special_token_ids"]["bos"], "eos_token_id": identity["special_token_ids"]["eos"], "pad_token_id": identity["special_token_ids"]["pad"], "unk_token_id": identity["special_token_ids"]["unk"], "tokenizer_artifact_sha256": identity["artifact_sha256"], "tokenizer_manifest_canonical_sha256": canonical_manifest, "tokenizer_semantic_identity_sha256": semantic_identity, "tokenization_behavior_sha256": behavior_hash, "selected_sequence_length": sequence_length, "response_only_masking": True, "train_optimizer_use": True, "validation_optimizer_use": False, "challenge_optimizer_use": False, "sanity_optimizer_use": False, "challenge_cached": False, "sanity_cached": False, "source_hashes": {name: sha256(V3_SFT / f"{name}.jsonl") for name in ("train", "validation")}, "splits": stats, "generation_tool_version": "genpy-sft-v3.1-token-cache-v1", "tokenization_behavior_vectors": vectors}
    path = ROOT / args.manifest; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: stats[name][key] for key in ("document_count", "stored_token_count", "assistant_token_count", "ignored_prompt_token_count", "truncation_count", "maximum_stored_input_positions")} for name in stats}, indent=2))


if __name__ == "__main__": main()
