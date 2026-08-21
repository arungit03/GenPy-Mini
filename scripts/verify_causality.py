"""Run final-logit and intermediate-layer causal-isolation checks."""

from __future__ import annotations

import torch

from _verification_cli import tiny_config
from genpy.model import GenPyForCausalLM
from genpy.verification.causal import causal_isolation, intermediate_layer_isolation


def main() -> int:
    model = GenPyForCausalLM(tiny_config(), attention_backend="eager").eval()
    first = torch.tensor([[10, 20, 30, 40, 50]])
    second = torch.tensor([[10, 20, 30, 99, 88]])
    final = causal_isolation(model, first, second, 3)
    layer = intermediate_layer_isolation(model, first, second, 3)
    passed = final["passed"] and layer["passed"]
    print(f"Final causal isolation: {'PASS' if final['passed'] else 'FAIL'} ({final['max_difference']:.3g})")
    print(f"Intermediate causal isolation: {'PASS' if layer['passed'] else 'FAIL'} ({layer['max_difference']:.3g})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
