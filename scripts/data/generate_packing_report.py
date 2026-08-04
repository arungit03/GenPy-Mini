"""Render the machine packing report as concise Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.training.packing import load_packing_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_packing_config(args.config, ROOT)
    report_path = ROOT / str(config.packing["report_json"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lines = [
        "# Packing Report", "", f"Mode: `{report['mode']}`", "",
        "| Group | Records | Samples | Real tokens | Padding | Active targets |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, group in report["groups"].items():
        lines.append(
            f"| {name} | {group.get('input_record_count', 0)} | "
            f"{group.get('packed_sample_count', 0)} | {group.get('real_tokens', 0)} | "
            f"{group.get('padding_tokens', 0)} | {group.get('active_loss_targets', 0)} |"
        )
    output = ROOT / str(config.packing["report_markdown"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "groups": len(report["groups"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
