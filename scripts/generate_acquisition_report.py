"""Generate JSON and Markdown dataset acquisition audit reports.

Reads whatever manifests exist under ``--manifest-dir`` and writes both a
machine-readable JSON report and a human-readable Markdown report. If the
dataset source registry and license policy can also be loaded (from their
default ``config/`` locations, or ``--registry`` / ``--license-policy``),
the report additionally includes registered/enabled/approved/review
-required/rejected counts; otherwise those are omitted with a warning,
since they cannot be recovered from manifests alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for _path in (_PROJECT_ROOT, _SRC_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from config.settings import CONFIG_DIR, DATA_DIR  # noqa: E402
from genpy.data.exceptions import DatasetGovernanceError  # noqa: E402
from genpy.data.licenses import LicensePolicy, load_license_policy  # noqa: E402
from genpy.data.reporting import (  # noqa: E402
    build_report,
    write_json_report,
    write_markdown_report,
)
from genpy.data.schemas import DatasetSourceRegistry  # noqa: E402
from genpy.data.source_registry import load_source_registry  # noqa: E402

DEFAULT_REGISTRY_PATH = CONFIG_DIR / "dataset_sources.yaml"
DEFAULT_LICENSE_POLICY_PATH = CONFIG_DIR / "license_policy.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DATA_DIR / "manifests")
    parser.add_argument(
        "--output-json", type=Path, default=DATA_DIR / "reports" / "acquisition-report.json"
    )
    parser.add_argument(
        "--output-markdown", type=Path, default=DATA_DIR / "reports" / "acquisition-report.md"
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--license-policy", type=Path, default=DEFAULT_LICENSE_POLICY_PATH)
    parser.add_argument(
        "--sources-root",
        type=Path,
        default=None,
        help="Defaults to <manifest-dir>/../raw/sources, matching the standard data/ layout.",
    )
    return parser.parse_args(argv)


def _load_registry_and_policy(
    registry_path: Path, policy_path: Path
) -> tuple[DatasetSourceRegistry | None, LicensePolicy | None]:
    try:
        registry = load_source_registry(registry_path)
        policy = load_license_policy(policy_path)
    except DatasetGovernanceError as exc:
        print(
            f"Warning: could not load registry/license policy ({exc}); "
            "report will omit registration-based counts.",
            file=sys.stderr,
        )
        return None, None
    return registry, policy


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    registry, policy = _load_registry_and_policy(args.registry, args.license_policy)
    report = build_report(
        args.manifest_dir, sources_root=args.sources_root, registry=registry, policy=policy
    )

    try:
        write_json_report(report, args.output_json)
        write_markdown_report(report, args.output_markdown)
    except OSError as exc:
        print(f"Could not write report: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_markdown}")
    print(f"Acquired sources: {report.acquired_sources}, total bytes: {report.total_bytes:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
