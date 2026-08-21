"""Run the CUDA-only canonical 1024-token verification."""

from __future__ import annotations

import json
import sys

from verify_gpu import ROOT, run_gpu_verification
from genpy.config import load_config


def main() -> int:
    result = run_gpu_verification(load_config(ROOT / "configs/model_200m.yaml"), full_context=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/checkpoint_5_gpu_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    status = result["full_context"]["status"]
    print(f"Full context 1024: {status}")
    return 0 if status in {"PASS", "NOT_RUN", "SKIPPED_NO_CUDA"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
