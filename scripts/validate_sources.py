"""Validate the dataset source registry and evaluate license/governance status.

Run with ``python scripts/validate_sources.py --all`` (or ``--source <id>``
for a single source). This script never acquires anything -- see
``scripts/acquire_sources.py`` for that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for _path in (_PROJECT_ROOT, _SRC_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from config.settings import CONFIG_DIR  # noqa: E402
from genpy.data.exceptions import DatasetGovernanceError  # noqa: E402
from genpy.data.licenses import load_license_policy  # noqa: E402
from genpy.data.source_registry import (  # noqa: E402
    SourceEvaluation,
    evaluate_registry,
    load_source_registry,
)

DEFAULT_REGISTRY_PATH = CONFIG_DIR / "dataset_sources.yaml"
DEFAULT_LICENSE_POLICY_PATH = CONFIG_DIR / "license_policy.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="Path to dataset_sources.yaml"
    )
    parser.add_argument(
        "--license-policy",
        type=Path,
        default=DEFAULT_LICENSE_POLICY_PATH,
        help="Path to license_policy.yaml",
    )
    parser.add_argument(
        "--source", metavar="SOURCE_ID", default=None, help="Validate only this source id"
    )
    parser.add_argument(
        "--all", action="store_true", help="Validate every registered source (default)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON instead of text"
    )
    return parser.parse_args(argv)


def _evaluation_to_dict(evaluation: SourceEvaluation) -> dict[str, object]:
    return {
        "id": evaluation.source.id,
        "enabled": evaluation.source.enabled,
        "source_type": evaluation.source.source_type,
        "declared_spdx": evaluation.source.license.declared_spdx,
        "license_status": evaluation.license_status,
        "governance_status": evaluation.source.governance.approval_status,
        "effective_status": evaluation.effective_status,
        "reasons": list(evaluation.reasons),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        registry = load_source_registry(args.registry)
        policy = load_license_policy(args.license_policy)
    except DatasetGovernanceError as exc:
        print(f"Configuration is invalid: {exc}", file=sys.stderr)
        return 1

    evaluations = list(evaluate_registry(registry, policy))

    if args.source:
        evaluations = [e for e in evaluations if e.source.id == args.source]
        if not evaluations:
            print(f"Source {args.source!r} is not registered in {args.registry}.", file=sys.stderr)
            return 1

    if args.json:
        print(json.dumps([_evaluation_to_dict(e) for e in evaluations], indent=2))
        return 0

    print(f"Registry: {args.registry}")
    print(f"License policy: {args.license_policy}")
    print(f"Sources evaluated: {len(evaluations)}")
    print("-" * 72)
    for evaluation in evaluations:
        source = evaluation.source
        print(
            f"[{evaluation.effective_status.upper():^15}] {source.id} "
            f"(license={source.license.declared_spdx} -> {evaluation.license_status}, "
            f"enabled={source.enabled})"
        )
        for reason in evaluation.reasons:
            print(f"    - {reason}")

    approved = sum(1 for e in evaluations if e.effective_status == "approved")
    review_required = sum(1 for e in evaluations if e.effective_status == "review_required")
    rejected = sum(1 for e in evaluations if e.effective_status == "rejected")
    print("-" * 72)
    print(f"approved={approved} review_required={review_required} rejected={rejected}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
