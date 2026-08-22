import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v31_common import canonical_sha256, tokenizer_identity


def test_canonical_tokenizer_hash_and_identity():
    manifest, identity, manifest_hash, semantic_hash = tokenizer_identity()
    assert manifest_hash == json.loads((ROOT / "reports/checkpoint_8_v3_1/tokenizer_identity.json").read_text())["tokenizer_manifest_canonical_sha256"]
    assert semantic_hash == json.loads((ROOT / "reports/checkpoint_8_v3_1/tokenizer_identity.json").read_text())["tokenizer_semantic_identity_sha256"]
    assert canonical_sha256(identity) == semantic_hash


def test_tokenization_behavior_fingerprint_is_stable():
    report = json.loads((ROOT / "reports/checkpoint_8_v3_1/tokenizer_identity.json").read_text())
    assert len(report["tokenization_behavior"]["vectors"]) == 4
    assert len(report["tokenization_behavior_sha256"]) == 64


def test_sequence_selection_excludes_frozen_evaluation_splits():
    report = json.loads((ROOT / "reports/checkpoint_8_v3_1/sequence_length_analysis.json").read_text())
    assert report["selection_sources"] == ["train", "validation"]
    assert report["challenge_used_for_sequence_selection"] is False
    assert report["sanity_used_for_sequence_selection"] is False
    selected = str(report["selected_sequence_length"])
    assert report["candidates"][selected]["train"]["truncation_count"] == 0
    assert report["candidates"][selected]["validation"]["truncation_count"] == 0


def test_cache_integrity_and_evaluation_cache_absence():
    manifest = json.loads((ROOT / "data/instruction/tokenized_v3/SFT_V3_TOKEN_CACHE_MANIFEST.json").read_text())
    cache = ROOT / "data/instruction/tokenized_v3"
    assert set(manifest["splits"]) == {"train", "validation"}
    for name in ("train", "validation"):
        offsets = np.load(cache / f"{name}.offsets.npy", mmap_mode="r")
        assert offsets[0] == 0
        assert np.all(np.diff(offsets) >= 0)
        assert manifest["splits"][name]["truncation_count"] == 0
    assert not any((cache / f"{name}.{suffix}").exists() for name in ("challenge", "sanity") for suffix in ("input_ids.bin", "labels.bin", "offsets.npy"))


def test_response_only_masking_and_ranges():
    cache = ROOT / "data/instruction/tokenized_v3"
    for name in ("train", "validation"):
        inputs = np.memmap(cache / f"{name}.input_ids.bin", mode="r", dtype=np.uint16)
        labels = np.memmap(cache / f"{name}.labels.bin", mode="r", dtype=np.int32)
        assert int(inputs.min()) >= 0 and int(inputs.max()) < 32000
        assert np.all((labels == -100) | ((labels >= 0) & (labels < 32000)))
        assert np.any(labels == -100) and np.any(labels >= 0)


def test_budget_math_and_config_contract():
    budget = json.loads((ROOT / "reports/checkpoint_8_v3_1/sft_budget.json").read_text())
    assert budget["train_examples"] == 3000
    assert budget["micro_batch_size"] == 1 and budget["gradient_accumulation_steps"] == 8
    assert budget["updates_per_dataset_pass"] == 375
    assert budget["recommended_dataset_passes"] == 3
    assert budget["recommended_global_max_steps"] == 1125
    assert json.loads((ROOT / "reports/checkpoint_8_v3_1/preflight_local.json").read_text())["optimizer_steps_performed"] == 0


def test_local_preflight_and_cache_reproducibility_pass():
    preflight = json.loads((ROOT / "reports/checkpoint_8_v3_1/preflight_local.json").read_text())
    reproducibility = json.loads((ROOT / "reports/checkpoint_8_v3_1/cache_reproducibility.json").read_text())
    assert preflight["status"] == "READY_FOR_KAGGLE"
    assert preflight["static_gates_pass"] is True
    assert reproducibility["cache_reproducibility_pass"] is True


def test_v3_source_hashes_remain_frozen():
    readiness = json.loads((ROOT / "reports/checkpoint_8_v3_1/READINESS.json").read_text())
    assert readiness["source_immutable_after"] is True
    assert readiness["model_weights_modified"] is False
    assert readiness["tokenizer_modified"] is False
    assert readiness["optimizer_steps"] == 0
