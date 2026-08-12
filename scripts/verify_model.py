"""Run CPU-safe Step 5 verification of the GenPy architecture."""

import sys
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import torch

from genpy.config import ModelConfig, load_model_config
from genpy.model import GenPyForCausalLM, RotaryEmbedding
from genpy.verification import causal_lm_loss, gradient_report
from genpy.verification.diagnostics import bias_free_linear_names


ROOT = Path(__file__).resolve().parents[1]


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


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> int:
    try:
        config = load_model_config(ROOT / "configs" / "model_200m.yaml")
        production = GenPyForCausalLM(config)
        production_count = sum(parameter.numel() for parameter in production.parameters())
        check(production_count == 201_560_832, "production parameter count")
        check(len(production.blocks) == 24, "production block count")
        check(config.hidden_size == 768 and config.num_heads == 12 and config.head_dim == 64, "attention dimensions")
        check(config.intermediate_size == 2176 and config.vocab_size == 32000, "SwiGLU and vocabulary dimensions")
        check(config.max_seq_len == 1024, "production context")
        check(production.lm_head.weight is production.token_embedding.weight, "weight tying")
        check(not bias_free_linear_names(production), "bias-free production projections")

        model = GenPyForCausalLM(tiny_config()).eval()
        input_ids = torch.randint(0, 256, (2, 16))
        logits = model(input_ids)
        check(logits.shape == (2, 16, 256), "tiny forward shape")
        check(bool(torch.isfinite(logits).all().item()), "tiny logits finite")
        loss = causal_lm_loss(logits, input_ids)
        check(bool(torch.isfinite(loss).item()), "causal LM loss finite")
        model.zero_grad(set_to_none=True)
        loss.backward()
        report = gradient_report(model)
        check(not report["missing"], "all trainable gradients present")
        check(not report["non_finite"], "all gradients finite")

        model.eval()
        prefix = torch.randint(0, 256, (1, 4))
        first = torch.cat((prefix, torch.tensor([[5]])), dim=1)
        second = torch.cat((prefix, torch.tensor([[17]])), dim=1)
        with torch.no_grad():
            first_logits = model(first)
            second_logits = model(second)
        earlier_difference = (first_logits[:, :4] - second_logits[:, :4]).abs().max().item()
        changed_difference = (first_logits[:, 4] - second_logits[:, 4]).abs().max().item()
        check(earlier_difference < 1e-6, "causal isolation for earlier positions")
        check(changed_difference > 0.0, "current position responds to changed token")
        rope = RotaryEmbedding(16, 32)
        check(not list(rope.parameters()), "RoPE has no trainable parameters")
        print(f"INFO: causal earlier max difference={earlier_difference:.9g}")
        print(f"INFO: causal changed-position max difference={changed_difference:.9g}")
        print("CPU model verification completed successfully.")
        return 0
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
