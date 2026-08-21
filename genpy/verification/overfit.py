"""Bounded tiny-model memorization check; not a production training loop."""

from __future__ import annotations

from dataclasses import replace

import torch

from genpy.config import ModelConfig
from genpy.model import GenPyForCausalLM

from .loss import causal_lm_loss


def tiny_overfit_config(config: ModelConfig) -> ModelConfig:
    return replace(config, vocab_size=128, max_seq_len=32, n_layers=2, d_model=64, n_heads=4, head_dim=16, ffn_hidden_size=128)


def run_tiny_overfit(config: ModelConfig, seed: int = 42, steps: int = 200, learning_rate: float = 0.003) -> dict:
    torch.manual_seed(seed)
    model = GenPyForCausalLM(tiny_overfit_config(config), attention_backend="eager")
    model.train()
    # Each row has a distinct first token; otherwise identical BOS-only context
    # makes the first target intrinsically ambiguous and imposes a loss floor.
    sequences = torch.tensor([[1, 10, 11, 12, 2], [20, 21, 22, 23, 2], [30, 31, 32, 33, 2], [40, 41, 42, 43, 2]], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    initial_loss = None
    minimum_loss = float("inf")
    final_loss = float("inf")
    checkpoints: list[float] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = causal_lm_loss(model(sequences), sequences)
        if initial_loss is None:
            initial_loss = float(loss.detach().item())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
        minimum_loss = min(minimum_loss, final_loss)
        if step in {0, steps // 4, steps // 2, steps - 1}:
            checkpoints.append(final_loss)
    reduction = 100.0 * (initial_loss - final_loss) / initial_loss if initial_loss else 0.0
    passed = final_loss < 0.1 or reduction >= 95.0
    return {"seed": seed, "steps": steps, "initial_loss": initial_loss, "final_loss": final_loss, "minimum_loss": minimum_loss, "loss_reduction_percent": reduction, "checkpoints": checkpoints, "passed": passed}
