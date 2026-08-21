"""Gradient coverage helpers."""

from __future__ import annotations

from torch import nn


def representative_gradient_names(model: nn.Module) -> dict[str, bool]:
    checks = {key: False for key in ("embedding", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "rmsnorm", "final_norm")}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("token_embedding"):
            checks["embedding"] = True
        for key in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            if f".{key}.weight" in name:
                checks[key] = True
        if ".attn_norm.weight" in name or ".ffn_norm.weight" in name:
            checks["rmsnorm"] = True
        if name == "final_norm.weight":
            checks["final_norm"] = True
    return checks
