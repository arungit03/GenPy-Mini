"""Plan or explicitly stage optional Python sources without implicit scraping."""

from pathlib import Path
import argparse
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["codesearchnet"], required=True)
    parser.add_argument("--source-path", type=Path, help="Existing local archive/file; no URL scraping is performed")
    parser.add_argument("--destination", type=Path, default=Path("data/production/raw/codesearchnet"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    expected = "CodeSearchNet Python subset with explicit per-record license metadata"
    print(f"Source: {args.source}")
    print(f"Expected source: {expected}")
    print(f"Destination: {args.destination}")
    if args.dry_run:
        print("Dry run: no files downloaded or copied.")
        return 0
    if args.source_path is None:
        print("No acquisition URL is used automatically. Supply --source-path for a reviewed local copy.", file=sys.stderr)
        return 2
    if not args.source_path.exists():
        print(f"Source path does not exist: {args.source_path}", file=sys.stderr)
        return 2
    if args.destination.exists() and any(args.destination.iterdir()) and not args.force:
        print("Destination is non-empty; pass --force to overwrite it.", file=sys.stderr)
        return 2
    args.destination.mkdir(parents=True, exist_ok=True)
    target = args.destination / args.source_path.name
    if args.source_path.is_dir():
        shutil.copytree(args.source_path, args.destination, dirs_exist_ok=True)
    else:
        shutil.copy2(args.source_path, target)
    print(f"Staged reviewed local source at: {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
