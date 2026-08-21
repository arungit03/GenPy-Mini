"""Safely restore a packaged Checkpoint 7 training archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_members(archive: tarfile.TarFile) -> None:
    for member in archive.getmembers():
        name = Path(member.name)
        if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
            raise ValueError(f"unsafe archive member: {member.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sha256", default=None)
    args = parser.parse_args()
    archive_path = Path(args.archive)
    if args.sha256:
        expected = Path(args.sha256).read_text(encoding="utf-8").split()[0]
        actual = sha256(archive_path)
        if actual != expected:
            raise ValueError("archive SHA256 mismatch")
    temporary = Path(tempfile.mkdtemp(prefix="genpy-restore-"))
    try:
        with tarfile.open(archive_path, mode="r") as archive:
            validate_members(archive)
            archive.extractall(temporary)
        pointer = temporary / "checkpoints/latest.json"
        if not pointer.is_file():
            raise ValueError("archive has no latest.json")
        checkpoint_name = json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
        checkpoint = temporary / "checkpoints" / checkpoint_name
        if not (checkpoint / "COMPLETE").is_file():
            raise ValueError("archive latest checkpoint has no COMPLETE marker")
        destination = Path(args.run_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for source in temporary.iterdir():
            target = destination / source.name
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
        restored_pointer = destination / "checkpoints/latest.json"
        resolved = destination / "checkpoints" / json.loads(restored_pointer.read_text(encoding="utf-8"))["checkpoint"]
        if not (resolved / "COMPLETE").is_file():
            raise ValueError("restored latest checkpoint does not resolve")
        print(f"Restored {resolved} to {destination}")
        return 0
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
