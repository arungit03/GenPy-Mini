"""Report Phase 4 model and production-packing readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.model.config import load_model_config  # noqa: E402
from genpy.model.readiness import check_model_readiness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_model_config(args.config, ROOT)
    result = check_model_readiness(args.config, ROOT)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if config.is_smoke:
        return 0 if result.status == "READY_FOR_SMOKE_MODEL" else 1
    return 0 if result.status in {"READY_FOR_PACKING", "READY_FOR_PHASE5"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
