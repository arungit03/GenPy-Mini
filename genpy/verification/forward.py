"""Forward-pass verification helpers."""

from __future__ import annotations

import torch
from torch import nn

from .numerical import assert_finite


def run_forward_shapes(model: nn.Module, vocab_size: int, device: torch.device | str = "cpu") -> dict:
    model = model.to(device).eval()
    records = []
    for batch, sequence in ((1, 1), (1, 8), (2, 16)):
        ids = torch.randint(0, vocab_size, (batch, sequence), device=device)
        with torch.no_grad():
            logits = model(ids)
        assert_finite(logits, f"logits[{batch},{sequence}]")
        records.append({"batch": batch, "sequence": sequence, "shape": list(logits.shape), "dtype": str(logits.dtype), "device": str(logits.device)})
    return {"passed": True, "cases": records}
