"""Deep causal-isolation checks."""

from __future__ import annotations

import torch
from torch import nn


def causal_isolation(model: nn.Module, first: torch.Tensor, second: torch.Tensor, prefix_length: int) -> dict:
    model.eval()
    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)
    difference = float((first_logits[:, :prefix_length] - second_logits[:, :prefix_length]).abs().max().item())
    return {"max_difference": difference, "prefix_length": prefix_length, "passed": difference <= 1e-5}


def intermediate_layer_isolation(model: nn.Module, first: torch.Tensor, second: torch.Tensor, prefix_length: int, layer_index: int = 0) -> dict:
    captured: list[torch.Tensor] = []
    handle = model.layers[layer_index].register_forward_hook(lambda _module, _inputs, output: captured.append(output.detach()))
    try:
        model.eval()
        with torch.no_grad():
            model(first)
            first_hidden = captured.pop()
            model(second)
            second_hidden = captured.pop()
    finally:
        handle.remove()
    difference = float((first_hidden[:, :prefix_length] - second_hidden[:, :prefix_length]).abs().max().item())
    return {"layer": layer_index, "max_difference": difference, "prefix_length": prefix_length, "passed": difference <= 1e-5}
