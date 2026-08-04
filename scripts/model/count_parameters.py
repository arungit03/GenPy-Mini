"""Print an allocation-free exact GenPy parameter audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.model.config import load_model_config  # noqa: E402
from genpy.model.parameter_count import count_parameters, validate_declared_tier  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_model_config(args.config, ROOT)
    audit = count_parameters(config)
    validate_declared_tier(config, audit)
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
