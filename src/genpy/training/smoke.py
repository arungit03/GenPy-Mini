"""Bounded CPU-only forward/backward and micro-overfit correctness checks."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, cast

import torch

from genpy.model.block import TransformerBlock
from genpy.model.config import load_model_config
from genpy.model.transformer import build_model
from genpy.training.packed_dataset import PackedDataset
from genpy.training.packing import load_packing_config, prepare_packed_data


def _smoke_batch(model_config_path: Path, data_config_path: Path) -> tuple[Any, dict[str, Any]]:
    model_config = load_model_config(model_config_path)
    data_config = load_packing_config(data_config_path, model_config.project_root)
    prepare_packed_data(data_config)
    manifest = data_config.output_root / "manifests/packing_manifest.json"
    dataset = PackedDataset(
        manifest,
        family="pretraining",
        split="train",
        tokenizer_fingerprint=str(data_config.tokenizer["fingerprint"]),
        packing_configuration_hash=data_config.config_hash,
    )
    sample = dataset[0]
    batch = {
        "input_ids": sample.input_ids.unsqueeze(0),
        "labels": sample.labels.unsqueeze(0),
        "attention_mask": sample.attention_mask.unsqueeze(0),
    }
    return model_config, batch


def smoke_forward(model_config_path: Path, data_config_path: Path) -> dict[str, Any]:
    """Run one deterministic CPU float32 forward, backward, and optimizer step."""
    config, batch = _smoke_batch(model_config_path, data_config_path)
    if not config.is_smoke:
        raise ValueError("forward smoke command requires the smoke model")
    model = build_model(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first_block = cast(TransformerBlock, model.blocks[0])
    before = first_block.attention.query.weight.detach().clone()
    output = model(**batch)
    if output.loss is None or not bool(torch.isfinite(output.loss)):
        raise RuntimeError("smoke loss is missing or non-finite")
    output.loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise RuntimeError("smoke gradients are non-finite")
    optimizer.step()
    changed = not torch.equal(before, first_block.attention.query.weight.detach())
    if not changed:
        raise RuntimeError("optimizer step did not change an expected parameter")
    return {
        "device": "cpu",
        "dtype": "float32",
        "logits_shape": list(output.logits.shape),
        "loss": float(output.loss.detach()),
        "active_targets": output.token_count,
        "gradients_finite": finite_gradients,
        "parameter_changed": changed,
    }


def micro_overfit(
    model_config_path: Path,
    data_config_path: Path,
    *,
    maximum_steps: int,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Overfit one safe fixture batch under hard step and wall-time limits."""
    if not 2 <= maximum_steps <= 100:
        raise ValueError("micro-overfit steps must be between 2 and 100")
    config, batch = _smoke_batch(model_config_path, data_config_path)
    if not config.is_smoke:
        raise ValueError("micro-overfit requires the smoke model")
    model = build_model(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial_loss = final_loss = math.inf
    started = time.perf_counter()
    completed = 0
    for step in range(maximum_steps):
        if time.perf_counter() - started > timeout_seconds:
            raise TimeoutError("micro-overfit exceeded its wall-time limit")
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        assert output.loss is not None
        if not bool(torch.isfinite(output.loss)):
            raise RuntimeError("micro-overfit loss became non-finite")
        loss_value = float(output.loss.detach())
        if step == 0:
            initial_loss = loss_value
        final_loss = loss_value
        output.loss.backward()
        optimizer.step()
        completed += 1
    if final_loss >= initial_loss * 0.8:
        raise RuntimeError("micro-overfit loss did not decrease materially")
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "steps": completed,
        "loss_reduction_percentage": 100.0 * (initial_loss - final_loss) / initial_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "safe fixture correctness only; not model-quality evidence",
    }
