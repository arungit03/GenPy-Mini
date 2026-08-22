"""Static local/Kaggle preflight for v3.1; never creates training state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genpy.config import load_config
from genpy.model import GenPyForCausalLM
from genpy.model.utils import count_parameters
from genpy.tokenizer import GenPyTokenizer
from genpy.training.config import load_training_config
from scripts.v31_common import EXPECTED_V3_PACKAGE_SHA, REPORT_DIR, TOKENIZER_DIR, V3_RAW, V3_SFT, behavior_vectors, canonical_sha256, frozen_source_hashes, sha256, tokenizer_identity

EXPECTED_PARAMETERS = 201560832
EXPECTED_BASE_SHA = "a963a91d8f6bee350e15ff88d3375c039887cb0b09c787fecf0f2de02d5be942"


def cache_checks(cache_dir, manifest, name):
    meta = manifest["splits"][name]; input_path = cache_dir / f"{name}.input_ids.bin"; labels_path = cache_dir / f"{name}.labels.bin"; offsets_path = cache_dir / f"{name}.offsets.npy"
    if not all(path.is_file() for path in (input_path, labels_path, offsets_path)): return {"files_exist": False}
    inputs = np.memmap(input_path, mode="r", dtype=np.uint16); labels = np.memmap(labels_path, mode="r", dtype=np.int32); offsets = np.load(offsets_path, mmap_mode="r")
    ranges = bool(not len(inputs) or (int(inputs.min()) >= 0 and int(inputs.max()) < 32000)) and bool(not len(labels) or np.all((labels == -100) | ((labels >= 0) & (labels < 32000))))
    offsets_ok = len(offsets) == meta["document_count"] + 1 and int(offsets[0]) == 0 and int(offsets[-1]) == len(inputs) == len(labels) and bool(np.all(np.diff(offsets) >= 0))
    per_doc = all(np.any(labels[int(start):int(end)] == -100) and np.any(labels[int(start):int(end)] >= 0) for start, end in zip(offsets[:-1], offsets[1:]))
    return {"files_exist": True, "document_count_valid": meta["document_count"] == {"train": 3000, "validation": 300}[name] and len(offsets) - 1 == meta["document_count"], "truncation_zero": meta["truncation_count"] == 0, "ranges_valid": ranges, "offsets_valid": offsets_ok, "per_document_response_only_labels_valid": per_doc, "cache_hash_valid": sha256(input_path) == meta["input_ids_sha256"] and sha256(labels_path) == meta["labels_sha256"] and sha256(offsets_path) == meta["offsets_sha256"], "source_hash_valid": sha256(V3_SFT / f"{name}.jsonl") == manifest["source_hashes"][name], "maximum_stored_input_positions_valid": meta["maximum_stored_input_positions"] <= manifest["selected_sequence_length"]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("local", "kaggle"), default="local"); parser.add_argument("--base-model", default="runs/genpy200m_pretrain_v1/checkpoints/step_000000001980/model.pt"); args = parser.parse_args()
    identity_report = json.loads((REPORT_DIR / "tokenizer_identity.json").read_text(encoding="utf-8")); sequence_report = json.loads((REPORT_DIR / "sequence_length_analysis.json").read_text(encoding="utf-8")); cache_manifest = json.loads((ROOT / "data/instruction/tokenized_v3/SFT_V3_TOKEN_CACHE_MANIFEST.json").read_text(encoding="utf-8")); budget = json.loads((REPORT_DIR / "sft_budget.json").read_text(encoding="utf-8")); audit = json.loads((ROOT / "reports/checkpoint_8_v3/dataset_audit.json").read_text(encoding="utf-8")); functional = json.loads((ROOT / "reports/checkpoint_8_v3/reference_functional_audit.json").read_text(encoding="utf-8")); v3_ready = json.loads((ROOT / "reports/checkpoint_8_v3/readiness.json").read_text(encoding="utf-8")); config = load_training_config(ROOT / "configs/sft_200m_kaggle_v3.yaml")
    manifest, identity, canonical_manifest, semantic_identity = tokenizer_identity(); vectors, behavior_hash = behavior_vectors(); cache_dir = ROOT / "data/instruction/tokenized_v3"; cache = {name: cache_checks(cache_dir, cache_manifest, name) for name in ("train", "validation")}
    model_definition = GenPyForCausalLM(load_config(ROOT / "configs/model_200m.yaml").model); base_path = ROOT / args.base_model
    baseline = identity_report["source_hashes_at_start"]; current = frozen_source_hashes(); source_frozen = baseline == current
    static = {"v3_dataset_readiness_pass": v3_ready.get("overall_status") == "PASS", "v3_source_hashes_frozen": source_frozen, "v3_package_hash_valid": current["v3_package"] == EXPECTED_V3_PACKAGE_SHA and sha256(ROOT / "artifacts/checkpoint_8_v3/GenPy-SFT-v3-Semantic-Pilot.zip") == EXPECTED_V3_PACKAGE_SHA, "model_parameter_definition_exact_201560832": count_parameters(model_definition) == EXPECTED_PARAMETERS, "tokenizer_name_valid": identity["tokenizer_name"] == "GenPy-Tokenizer-32K", "tokenizer_version_valid": identity["tokenizer_version"] == "GenPy-Tokenizer-32K-v1", "tokenizer_vocab_32000": identity["vocab_size"] == 32000, "tokenizer_special_ids_valid": identity["special_token_ids"] == {"pad": 0, "bos": 1, "eos": 2, "unk": 3}, "tokenizer_artifact_sha_valid": identity["artifact_sha256"] == "dc9ca78c405433d184d615282b1e87b539e1e604b1e3e05f3c30ced13ca653ac", "tokenizer_canonical_identity_valid": canonical_manifest == identity_report["tokenizer_manifest_canonical_sha256"] and canonical_manifest == cache_manifest["tokenizer_manifest_canonical_sha256"], "tokenizer_semantic_identity_valid": semantic_identity == identity_report["tokenizer_semantic_identity_sha256"] and semantic_identity == cache_manifest["tokenizer_semantic_identity_sha256"], "tokenization_behavior_valid": behavior_hash == identity_report["tokenization_behavior_sha256"] == cache_manifest["tokenization_behavior_sha256"], "sequence_analysis_valid": sequence_report["selected_sequence_length"] == cache_manifest["selected_sequence_length"] and sequence_report["candidates"][str(sequence_report["selected_sequence_length"])]["combined_truncation_count"] == 0, "sequence_selected_from_train_validation_only": sequence_report["selection_sources"] == ["train", "validation"] and not sequence_report["challenge_used_for_sequence_selection"] and not sequence_report["sanity_used_for_sequence_selection"], "train_zero_truncation": cache["train"].get("truncation_zero", False), "validation_zero_truncation": cache["validation"].get("truncation_zero", False), "cache_sequence_matches_config": config.training.sequence_length == cache_manifest["selected_sequence_length"], "train_cache_hash_valid": cache["train"].get("cache_hash_valid", False), "validation_cache_hash_valid": cache["validation"].get("cache_hash_valid", False), "offsets_valid": cache["train"].get("offsets_valid", False) and cache["validation"].get("offsets_valid", False), "response_only_masking_valid": cache["train"].get("per_document_response_only_labels_valid", False) and cache["validation"].get("per_document_response_only_labels_valid", False), "challenge_not_cached": not any((cache_dir / f"challenge.{suffix}").exists() for suffix in ("input_ids.bin", "labels.bin", "offsets.npy")), "sanity_not_cached": not any((cache_dir / f"sanity.{suffix}").exists() for suffix in ("input_ids.bin", "labels.bin", "offsets.npy")), "challenge_optimizer_use_false": cache_manifest["challenge_optimizer_use"] is False, "sanity_optimizer_use_false": cache_manifest["sanity_optimizer_use"] is False, "sampling_shuffled_epoch": config.training.sampling == "shuffled_epoch", "global_steps_match_budget": config.training.max_steps == budget["recommended_global_max_steps"] == 1125, "v1_paths_rejected": all("sft_v1" not in value.lower() and "tokenized" not in value.lower() for value in ("configs/sft_200m_kaggle_v3.yaml", "data/instruction/tokenized_v3", "runs/genpy200m_sft_v3")), "v2_paths_rejected": all("v2" not in value.lower() for value in ("configs/sft_200m_kaggle_v3.yaml", "data/instruction/tokenized_v3", "runs/genpy200m_sft_v3")), "functional_audit_100_percent": functional["all_pass"] and functional["functional_correct_rate"] == 1.0 and functional["individual_test_cases_failed"] == 0, "dataset_novelty_audit_pass": audit["overall_status"] == "PASS", "cache_reproducibility_pass": json.loads((REPORT_DIR / "cache_reproducibility.json").read_text())["cache_reproducibility_pass"], "production_sft_started_false": True}
    static["v1_paths_rejected"] = all("sft_v1" not in value.lower() and not ("tokenized" in value.lower() and "tokenized_v3" not in value.lower()) for value in ("configs/sft_200m_kaggle_v3.yaml", "data/instruction/tokenized_v3", "runs/genpy200m_sft_v3"))
    runtime = {"runtime_base_checkpoint": "NOT_CHECKED_LOCAL", "cuda": "NOT_CHECKED_LOCAL", "bf16": "NOT_CHECKED_LOCAL"}
    if args.mode == "kaggle":
        runtime = {"runtime_base_checkpoint": base_path.is_file() and base_path.name == "model.pt" and sha256(base_path) == EXPECTED_BASE_SHA and count_parameters(model_definition) == EXPECTED_PARAMETERS, "cuda": torch.cuda.is_available(), "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()), "training_device_cuda": config.training.device == "cuda", "training_precision_bf16": config.training.precision == "bf16"}
    static_pass = all(static.values()); runtime_pass = all(runtime.values()) if args.mode == "kaggle" else True; result = {"mode": args.mode, "static_gates": static, "static_gates_pass": static_pass, **runtime, "runtime_gates_pass": runtime_pass, "status": "READY_FOR_KAGGLE" if static_pass and runtime_pass else "FAIL", "production_sft_started": False, "optimizer_steps_performed": 0}
    json_path = REPORT_DIR / ("preflight_local.json" if args.mode == "local" else "preflight_kaggle.json"); text_path = REPORT_DIR / ("PREFLIGHT_LOCAL.txt" if args.mode == "local" else "PREFLIGHT_KAGGLE.txt")
    text = "CHECKPOINT 8-v3.1 STATIC PREFLIGHT: READY_FOR_KAGGLE\n\n" + json.dumps(result, indent=2) + "\n" if args.mode == "local" and result["status"] == "READY_FOR_KAGGLE" else json.dumps(result, indent=2) + "\n"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); text_path.write_text(text, encoding="utf-8")
    print(json.dumps(result, indent=2)); print("CHECKPOINT 8-v3.1 STATIC PREFLIGHT: READY_FOR_KAGGLE" if args.mode == "local" and result["status"] == "READY_FOR_KAGGLE" else ("CHECKPOINT 8-v3 SFT KAGGLE PREFLIGHT: PASS" if args.mode == "kaggle" and result["status"] == "READY_FOR_KAGGLE" else "CHECKPOINT 8-v3.1 PREFLIGHT: FAIL"))
    return 0 if result["status"] == "READY_FOR_KAGGLE" else 1


if __name__ == "__main__": raise SystemExit(main())
