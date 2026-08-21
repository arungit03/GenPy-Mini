"""Run RMSNorm, RoPE, and finite-value checks."""

from __future__ import annotations

import torch

from _verification_cli import tiny_config
from genpy.model import RMSNorm, RotaryEmbedding
from genpy.verification.numerical import assert_finite, is_finite_tensor


def main() -> int:
    norm = RMSNorm(16)
    rms_pass = True
    for scale in (1.0, 1e-5, 1e3):
        value = norm(torch.randn(2, 4, 16) * scale)
        rms_pass = rms_pass and value.shape == (2, 4, 16) and is_finite_tensor(value)
    rope = RotaryEmbedding(16, 1024)
    q = torch.randn(1, 2, 1024, 16)
    rotated_q, rotated_k = rope(q, q.clone())
    rope_pass = rotated_q.shape == q.shape and rotated_k.shape == q.shape and is_finite_tensor(rotated_q)
    try:
        rope(torch.randn(1, 1, 1025, 16), torch.randn(1, 1, 1025, 16))
        boundary_pass = False
    except ValueError:
        boundary_pass = True
    try:
        assert_finite(torch.tensor([1.0, float("nan")]))
        deliberate_detection = False
    except ValueError:
        deliberate_detection = True
    passed = rms_pass and rope_pass and boundary_pass and deliberate_detection
    print(f"RMSNorm: {'PASS' if rms_pass else 'FAIL'}\nRoPE: {'PASS' if rope_pass else 'FAIL'}\nNon-finite detection: {'PASS' if deliberate_detection else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
