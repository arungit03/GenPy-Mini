"""AdamW construction with explicit decay/no-decay coverage audit."""

from __future__ import annotations

from torch import nn


def create_adamw(model: nn.Module, config) -> tuple[object, dict]:
    decay, no_decay = [], []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if "norm" in name.lower() or "embedding" in name.lower() or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    import torch
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.eps)
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    duplicate_count = len(optimizer_ids) - len(set(optimizer_ids))
    model_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer_id_set = set(optimizer_ids)
    audit = {"decay_tensors": len(decay), "no_decay_tensors": len(no_decay), "total_optimizer_parameters": sum(parameter.numel() for parameter in decay + no_decay), "duplicate_parameters": duplicate_count, "missing_parameters": len(model_ids - optimizer_id_set)}
    return optimizer, audit
