"""Verify causal next-token shifting against a reference implementation."""

from __future__ import annotations

import torch

from _verification_cli import tiny_config
from genpy.model import GenPyForCausalLM
from genpy.verification.loss import causal_lm_loss, reference_causal_lm_loss


def main() -> int:
    torch.manual_seed(7)
    model = GenPyForCausalLM(tiny_config(), attention_backend="eager").eval()
    ids = torch.randint(0, 128, (2, 8))
    logits = model(ids)
    actual = causal_lm_loss(logits, ids)
    reference = reference_causal_lm_loss(logits, ids)
    ignored = ids.clone(); ignored[:, 2] = -100
    ignored_loss = causal_lm_loss(logits, ignored, ignore_index=-100)
    difference = float((actual - reference).abs().item())
    passed = difference <= 1e-6 and torch.isfinite(ignored_loss)
    print(f"Causal shift: {'PASS' if passed else 'FAIL'}")
    print(f"Reference difference: {difference:.9g}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
