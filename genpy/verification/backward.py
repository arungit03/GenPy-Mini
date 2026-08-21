"""Backward-pass smoke helpers."""

from __future__ import annotations

import torch
from torch import nn

from .gradients import representative_gradient_names
from .loss import causal_lm_loss
from .numerical import gradient_audit


def run_backward_smoke(model: nn.Module, vocab_size: int, sequence_length: int = 8) -> dict:
    model.train()
    model.zero_grad(set_to_none=True)
    input_ids = torch.randint(0, vocab_size, (2, sequence_length), device=next(model.parameters()).device)
    logits = model(input_ids)
    loss = causal_lm_loss(logits, input_ids)
    loss.backward()
    audit = gradient_audit(model)
    names = representative_gradient_names(model)
    representative_pass = all(names.get(key, False) for key in ("embedding", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "rmsnorm", "final_norm"))
    return {"loss": float(loss.detach().item()), "gradient_audit": audit.to_dict(), "representative_gradients": names, "representative_pass": representative_pass, "passed": representative_pass and not audit.non_finite_gradients}
