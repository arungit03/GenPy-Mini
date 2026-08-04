from __future__ import annotations

import torch

from genpy.training.collator import build_packed_loader
from genpy.training.packed_dataset import PackedDataset
from genpy.training.packing import load_packing_config
from genpy.training.sampler import DeterministicSampler


def test_memory_mapped_dataset_returns_shifted_masked_tensors(phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    config = load_packing_config(phase4_fixture["packing_config"], phase4_fixture["root"])
    dataset = PackedDataset(
        config.output_root / "manifests/packing_manifest.json",
        family="pretraining",
        split="train",
        tokenizer_fingerprint=str(config.tokenizer["fingerprint"]),
        packing_configuration_hash=config.config_hash,
    )
    sample = dataset[0]
    assert sample.input_ids.shape == sample.labels.shape == torch.Size([16])
    assert sample.input_ids.dtype == sample.labels.dtype == torch.long
    assert sample.attention_mask.dtype == torch.bool
    assert len(dataset) > 1

    loader = build_packed_loader(
        dataset,
        DeterministicSampler(dataset, seed=42),
        batch_size=2,
        num_workers=0,
        seed=42,
    )
    batch = next(iter(loader))
    assert batch["input_ids"].shape == torch.Size([2, 16])
    assert batch["sample_indices"] == list(DeterministicSampler(dataset, seed=42))[:2]
