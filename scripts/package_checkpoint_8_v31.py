"""Build a deterministic v3.1 Kaggle preflight package."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def build_archive(output, paths, manifest_bytes):
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("PACKAGE_MANIFEST.json", date_time=(1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_bytes)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="artifacts/checkpoint_8_v3_1/GenPy-SFT-v3.1-Kaggle-Preflight.zip"); args = parser.parse_args()
    output = ROOT / args.output
    paths = []
    for directory in ("data/instruction/python_v3", "data/instruction/sft_v3", "data/instruction/tokenized_v3", "reports/checkpoint_8_v3", "reports/checkpoint_8_v3_1"):
        paths.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
    paths.extend([ROOT / "configs/sft_200m_kaggle_v3.yaml", ROOT / "configs/model_200m.yaml"])
    script_names = ("generate_sft_v3_semantic.py", "build_sft_v3_dataset.py", "audit_sft_v3_dataset.py", "verify_sft_v3_functional.py", "finalize_checkpoint_8_v3.py", "v31_common.py", "build_v3_tokenizer_identity.py", "analyze_sft_v3_sequence.py", "build_sft_v3_token_cache.py", "analyze_sft_v3_budget.py", "verify_sft_v3_cache_reproducibility.py", "preflight_sft_v3.py", "package_checkpoint_8_v31.py")
    paths.extend(ROOT / "scripts" / name for name in script_names)
    paths = sorted({path for path in paths if path.is_file() and path.name not in {"PACKAGE_MANIFEST.json", "GenPy-SFT-v3.1-Kaggle-Preflight.zip"} and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts})
    forbidden = [path for path in paths if any(token in str(path).lower() for token in ("model.pt", ".tar", "optimizer", "sft_v2", "tokenized_v2"))]
    if forbidden: raise RuntimeError("forbidden v3.1 package file: " + ", ".join(map(str, forbidden)))
    entries = [{"path": path.relative_to(ROOT).as_posix(), "size": path.stat().st_size, "sha256": sha(path)} for path in paths]
    manifest = {"format_version": 1, "package_name": "GenPy-SFT-v3.1-Kaggle-Preflight", "deterministic_zip": True, "fixed_zip_timestamp": "1980-01-01T00:00:00", "files": entries, "package_manifest_entry": "PACKAGE_MANIFEST.json (self-describing manifest excluded from its own hash list)", "excludes": ["model.pt", "CP7 tar files", "v2 model weights", "optimizer states", "training checkpoints", "temporary caches", ".git", "__pycache__", ".pytest_cache"]}
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    build_archive(output, paths, manifest_bytes)
    with tempfile.TemporaryDirectory(prefix="genpy-v31-package-") as temp:
        second = Path(temp) / output.name; build_archive(second, paths, manifest_bytes); reproducible = second.read_bytes() == output.read_bytes()
    manifest_path = output.parent / "PACKAGE_MANIFEST.json"; manifest_path.write_bytes(manifest_bytes); (output.parent / "GenPy-SFT-v3.1-Kaggle-Preflight.zip.sha256").write_text(f"{sha(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"package": str(output), "sha256": sha(output), "file_count": len(entries) + 1, "package_reproducibility_pass": reproducible}, indent=2))
    return 0 if reproducible else 1


if __name__ == "__main__": raise SystemExit(main())
