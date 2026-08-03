"""Streaming dataset statistics and honest smoke-report rendering."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def write_dataset_report(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    """Write machine-readable JSON and a concise Markdown dataset report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "dataset_report.json"
    markdown_path = reports_dir / "dataset_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    funnel = report.get("funnel", {})
    lines = [
        "# GenPy Dataset Report",
        "",
        f"Corpus version: `{report.get('corpus_version', 'unknown')}`",
        "",
        "This report contains smoke-test statistics unless explicitly marked otherwise.",
        "",
        "## Data Funnel",
        "",
        "| Stage | Records |",
        "| --- | ---: |",
    ]
    for stage in (
        "downloaded",
        "licence_accepted",
        "utf8_valid",
        "secret_pii_safe",
        "python_syntax_valid",
        "quality_accepted",
        "exact_deduplicated",
        "near_deduplicated",
        "split_and_sharded",
    ):
        lines.append(f"| {stage.replace('_', ' ').title()} | {funnel.get(stage, 0)} |")

    lines.extend(["", "## Summary", ""])
    summary_fields = (
        "total_records",
        "total_python_files",
        "valid_python_percentage",
        "downloaded_bytes",
        "extracted_bytes",
        "cleaned_bytes",
        "utf8_bytes",
        "characters",
        "lines",
        "python_lexical_tokens",
        "rough_subword_token_estimate",
        "peak_pipeline_memory_bytes",
        "processing_seconds",
    )
    for field in summary_fields:
        lines.append(f"- {field.replace('_', ' ').title()}: {report.get(field, 0)}")

    lines.extend(["", "## Rejections", ""])
    rejections = report.get("rejections", {})
    if rejections:
        for reason, count in sorted(rejections.items()):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "## Splits And Leakage", ""])
    for split, count in sorted(report.get("split_counts", {}).items()):
        lines.append(f"- {split}: {count}")
    leakage = report.get("leakage", {})
    for name, value in sorted(leakage.items()):
        lines.append(f"- {name.replace('_', ' ').title()}: {value}")

    lines.extend(
        [
            "",
            "## Remaining Risks",
            "",
            "- Static secret, PII, and safety scanners cannot guarantee complete removal.",
            "- Licence metadata and provenance still require human and legal review "
            "before a large run.",
            "- The token estimate is rough; exact counts wait for the GenPy tokenizer in Phase 3.",
            "- The smoke corpus is far below any model-training corpus target.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def increment(counter: Counter[str], name: str, amount: int = 1) -> None:
    """Typed helper for report counters."""
    counter[name] += amount
