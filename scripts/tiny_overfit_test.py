"""Verification-only tiny fixed-batch loss-decrease experiment."""

import sys
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import torch

from genpy.config import ModelConfig
from genpy.model import GenPyForCausalLM
from genpy.verification import causal_lm_loss


def tiny_config() -> ModelConfig:
    return ModelConfig(
        name="GenPy-verification-tiny",
        vocab_size=256,
        max_seq_len=32,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        intermediate_size=128,
        norm_eps=1e-5,
        rope_theta=10000.0,
        tie_embeddings=True,
    )


def main() -> int:
    torch.manual_seed(123)
    model = GenPyForCausalLM(tiny_config())
    model.train()
    input_ids = torch.randint(0, 256, (2, 16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    initial_loss = None
    final_loss = None
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = causal_lm_loss(model(input_ids), input_ids)
        if initial_loss is None:
            initial_loss = float(loss.detach().item())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    assert initial_loss is not None and final_loss is not None
    reduction = 100.0 * (initial_loss - final_loss) / initial_loss
    passed = final_loss < initial_loss * 0.25 and final_loss < 1.5
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
