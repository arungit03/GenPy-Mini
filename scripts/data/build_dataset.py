"""Build the bounded GenPy Phase 2 dataset pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.data.pipeline import DatasetPipeline  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--maximum-records", type=int)
    parser.add_argument("--maximum-download-size", type=int, dest="maximum_download_bytes")
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--confirm-large-download", action="store_true")
    return parser


def main() -> int:
    """Run the configured build while requiring confirmation for full-corpus mode."""
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.mode == "full" and not args.confirm_large_download:
        raise SystemExit(
            "Full mode requires --confirm-large-download after source, licence, "
            "size, and disk review."
        )
    pipeline = DatasetPipeline(
        args.config,
        project_root=ROOT,
        source_ids=set(args.sources) if args.sources else None,
        maximum_records=args.maximum_records,
        maximum_download_bytes=args.maximum_download_bytes,
        seed=args.seed,
        output_directory=args.output_directory,
        workers=args.workers,
        resume=args.resume,
    )
    report = pipeline.build(dry_run=args.dry_run)
    summary = {
        "corpus_version": report.get("corpus_version"),
        "dry_run": report.get("dry_run", False),
        "records": report.get("total_records", 0),
        "report": str(pipeline.paths["reports"] / "dataset_report.md"),
        "resumed": report.get("resumed_from_completed_run", False),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
