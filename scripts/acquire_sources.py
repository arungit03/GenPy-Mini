"""Acquire one or all approved dataset sources.

Exactly one of ``--source`` or ``--all-approved`` is required. A source
evaluated as ``rejected`` can never be acquired, with or without any flag.
A source evaluated as ``review_required`` can only be acquired when both
``--allow-review-required`` and a non-empty ``--override-reason`` are
given; the reason is recorded in that source's manifest.

Run with ``--dry-run`` first to see what would happen without writing
anything.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for _path in (_PROJECT_ROOT, _SRC_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from config.settings import CONFIG_DIR, DATA_DIR  # noqa: E402
from genpy.data.acquisition import acquire_source  # noqa: E402
from genpy.data.exceptions import DatasetGovernanceError  # noqa: E402
from genpy.data.licenses import load_license_policy  # noqa: E402
from genpy.data.schemas import DatasetSource  # noqa: E402
from genpy.data.source_registry import evaluate_registry, load_source_registry  # noqa: E402

DEFAULT_REGISTRY_PATH = CONFIG_DIR / "dataset_sources.yaml"
DEFAULT_LICENSE_POLICY_PATH = CONFIG_DIR / "license_policy.yaml"

_EXIT_OK = 0
_EXIT_ACQUISITION_FAILED = 1
_EXIT_USAGE_ERROR = 2
_EXIT_GOVERNANCE_BLOCKED = 3


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--license-policy", type=Path, default=DEFAULT_LICENSE_POLICY_PATH)
    parser.add_argument("--source", metavar="SOURCE_ID", default=None)
    parser.add_argument("--all-approved", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-review-required", action="store_true")
    parser.add_argument("--override-reason", metavar="TEXT", default=None)
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to data/raw")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if bool(args.source) == bool(args.all_approved):
        print("Exactly one of --source or --all-approved is required.", file=sys.stderr)
        return _EXIT_USAGE_ERROR

    try:
        registry = load_source_registry(args.registry)
        policy = load_license_policy(args.license_policy)
    except DatasetGovernanceError as exc:
        print(f"Configuration is invalid: {exc}", file=sys.stderr)
        return _EXIT_ACQUISITION_FAILED

    if args.source:
        source = registry.get(args.source)
        if source is None:
            print(f"Source {args.source!r} is not registered in {args.registry}.", file=sys.stderr)
            return _EXIT_ACQUISITION_FAILED
        candidates: tuple[DatasetSource, ...] = (source,)
    else:
        candidates = tuple(s for s in registry.sources if s.enabled)

    evaluations = {e.source.id: e for e in evaluate_registry(registry, policy)}

    output_root = args.output_root or (DATA_DIR / "raw")
    sources_root = output_root / "sources"
    downloads_root = output_root / "downloads"
    manifests_dir = DATA_DIR / "manifests"

    selected: list[DatasetSource] = []
    blocked_single_source = False

    for source in candidates:
        evaluation = evaluations[source.id]

        if evaluation.effective_status == "rejected":
            print(f"[REJECTED] {source.id}: {'; '.join(evaluation.reasons)} -- cannot be acquired.")
            blocked_single_source = True
            continue

        if evaluation.effective_status == "review_required":
            has_override = bool(
                args.allow_review_required and args.override_reason and args.override_reason.strip()
            )
            if not has_override:
                print(
                    f"[REVIEW REQUIRED] {source.id}: {'; '.join(evaluation.reasons)} -- "
                    "requires --allow-review-required and a non-empty --override-reason."
                )
                blocked_single_source = True
                continue

        selected.append(source)

    if not selected:
        print("No sources selected for acquisition.")
        return _EXIT_GOVERNANCE_BLOCKED if (args.source and blocked_single_source) else _EXIT_OK

    exit_code = _EXIT_OK
    for source in selected:
        evaluation = evaluations[source.id]
        governance_override = evaluation.effective_status == "review_required"

        print(f"\n=== {source.id} ({source.source_type}) ===")
        print(f"  revision: {source.revision}")
        print(f"  max download bytes: {source.acquisition.maximum_download_bytes:,}")
        print(f"  max extracted bytes: {source.acquisition.maximum_extracted_bytes:,}")
        if not source.enabled:
            print("  note: source is disabled in the registry but was explicitly selected")
        if governance_override:
            print(f"  governance override: allow-review-required, reason={args.override_reason!r}")

        try:
            outcome = acquire_source(
                source,
                sources_root=sources_root,
                downloads_root=downloads_root,
                manifests_dir=manifests_dir,
                force=args.force,
                dry_run=args.dry_run,
                governance_override=governance_override,
                override_reason=args.override_reason if governance_override else None,
                timeout_seconds=registry.defaults.timeout_seconds,
                retry_count=registry.defaults.retry_count,
            )
        except DatasetGovernanceError as exc:
            print(f"  FAILED: {exc}")
            exit_code = _EXIT_ACQUISITION_FAILED
            continue

        print(f"  status: {outcome.status}")
        if outcome.message:
            print(f"  {outcome.message}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
