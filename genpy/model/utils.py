"""Model inspection and parameter-counting utilities."""

from __future__ import annotations

from collections import OrderedDict

from torch import nn


def named_unique_parameters(model: nn.Module):
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            yield name, parameter


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for _, parameter in named_unique_parameters(model))


def parameter_summary(model: nn.Module) -> dict[str, int]:
    total = count_parameters(model)
    trainable = sum(parameter.numel() for _, parameter in named_unique_parameters(model) if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def parameter_breakdown(model: nn.Module) -> dict[str, int]:
    breakdown: OrderedDict[str, int] = OrderedDict((key, 0) for key in ("embeddings", "attention", "swiglu", "block_norms", "final_norm"))
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if name.startswith("token_embedding"):
            key = "embeddings"
        elif ".attention." in name:
            key = "attention"
        elif ".mlp." in name:
            key = "swiglu"
        elif ".attn_norm." in name or ".ffn_norm." in name:
            key = "block_norms"
        elif name.startswith("final_norm"):
            key = "final_norm"
        else:
            continue
        breakdown[key] += parameter.numel()
    return dict(breakdown)
