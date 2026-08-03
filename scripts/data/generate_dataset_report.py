"""Regenerate the readable report from verified machine-readable statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.data.statistics import write_dataset_report  # noqa: E402


def main() -> int:
    """Render Markdown without inventing or recomputing statistics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reports = ROOT / config["paths"]["reports"]
    report_path = reports / "dataset_report.json"
    if not report_path.exists():
        raise SystemExit("Run build_dataset.py before generating the report.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _, markdown = write_dataset_report(report, reports)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
