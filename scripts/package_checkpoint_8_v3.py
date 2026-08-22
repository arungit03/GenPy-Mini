"""Package only the v3 semantic dataset, scripts, and verification reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="artifacts/checkpoint_8_v3/GenPy-SFT-v3-Semantic-Pilot.zip"); args = parser.parse_args()
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for directory in ("data/instruction/python_v3", "data/instruction/sft_v3", "reports/checkpoint_8_v3"):
        paths.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
    paths.extend(ROOT / "scripts" / name for name in ("generate_sft_v3_semantic.py", "build_sft_v3_dataset.py", "audit_sft_v3_dataset.py", "verify_sft_v3_functional.py", "package_checkpoint_8_v3.py", "finalize_checkpoint_8_v3.py"))
    paths = sorted({path for path in paths if path.is_file()})
    forbidden = [path for path in paths if "v2" in str(path).lower() or path.suffix in {".pt", ".tar", ".pth"} or "optimizer" in str(path).lower()]
    if forbidden: raise RuntimeError("forbidden file in v3 package: " + ", ".join(map(str, forbidden)))
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths: archive.write(path, path.relative_to(ROOT).as_posix())
    listing = {"format_version": 3, "archive": str(output), "sha256": digest(output), "file_count": len(paths), "files": [path.relative_to(ROOT).as_posix() for path in paths], "excludes": ["CP7 model weights", "v2 weights", "optimizer states", "token cache", "production run directory"]}
    output.with_suffix(".manifest.json").write_text(json.dumps(listing, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(f"{listing['sha256']}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"package": str(output), "sha256": listing["sha256"], "file_count": len(paths)}, indent=2))


if __name__ == "__main__": main()
