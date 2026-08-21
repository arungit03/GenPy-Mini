"""Run the bounded tiny-dataset memorization check."""

from __future__ import annotations

import json

from _verification_cli import ROOT
from genpy.config import load_config
from genpy.verification.overfit import run_tiny_overfit


def main() -> int:
    result = run_tiny_overfit(load_config(ROOT / "configs/model_200m.yaml").model)
    report = ROOT / "reports/checkpoint_5_tiny_overfit.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Initial loss: {result['initial_loss']:.6f}")
    print(f"Final loss: {result['final_loss']:.6f}")
    print(f"Loss reduction: {result['loss_reduction_percent']:.2f}%")
    print(f"Tiny overfit: {'PASS' if result['passed'] else 'FAIL'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
