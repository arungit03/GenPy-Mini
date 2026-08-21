"""Run local forward and backward verification."""

from __future__ import annotations

import torch

from _verification_cli import ROOT, tiny_config
from genpy.config import load_config
from genpy.model import GenPyForCausalLM
from genpy.verification.backward import run_backward_smoke
from genpy.verification.forward import run_forward_shapes


def main() -> int:
    config = load_config(ROOT / "configs/model_200m.yaml")
    tiny = GenPyForCausalLM(tiny_config(), attention_backend="eager")
    forward = run_forward_shapes(tiny, 128)
    backward = run_backward_smoke(tiny, 128)
    production = GenPyForCausalLM(config.model).eval()
    with torch.no_grad():
        logits = production(torch.randint(0, config.model.vocab_size, (1, 4)))
    production_pass = tuple(logits.shape) == (1, 4, config.model.vocab_size) and bool(torch.isfinite(logits).all())
    print("Forward: PASS" if forward["passed"] and production_pass else "Forward: FAIL")
    print("Backward: PASS" if backward["passed"] else "Backward: FAIL")
    return 0 if forward["passed"] and production_pass and backward["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
