"""Prepare a bounded text corpus from a configured streaming source."""

import argparse
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from genpy.config import load_data_config
from genpy.data.pipeline import run_pipeline
from genpy.data.source import load_jsonl_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-jsonl", type=Path, help="Use a local JSONL source for offline smoke tests")
    args = parser.parse_args()
    config = load_data_config(args.config)
    print(f"Dataset: {config.dataset.name}")
    print(f"Configuration: {config.dataset.config}")
    print(f"Split: {config.dataset.split}")
    print(f"Streaming: {config.dataset.streaming}")
    print(f"Text field: {config.dataset.text_field}")
    print(f"Max documents: {args.max_documents if args.max_documents is not None else 'source limit'}")
    rows = load_jsonl_rows(args.source_jsonl) if args.source_jsonl else None
    result = run_pipeline(
        config,
        source=rows,
        max_documents=args.max_documents,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    print(f"Documents seen: {result.stats.source_documents_seen}")
    print(f"Accepted: {result.stats.accepted_documents}")
    print(f"Rejected: {result.stats.rejected_documents}")
    print(f"Duplicates: {result.stats.duplicate_documents}")
    print(f"Train: {result.stats.train_documents}")
    print(f"Validation: {result.stats.validation_documents}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Completed: {result.completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
