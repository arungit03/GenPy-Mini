"""Package only the auditable v2 data/reports for Kaggle transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/checkpoint_8_v2/GenPy-SFT-v2-Pilot.zip")
    args = parser.parse_args()
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True)
    sources = []
    for directory in ("data/instruction/python_v2", "data/instruction/sft_v2", "data/instruction/tokenized_v2", "reports/checkpoint_8_v2"):
        sources.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
    sources.extend([ROOT / "configs/sft_200m_kaggle_v2.yaml", ROOT / "scripts/generate_sft_v2_pilot.py", ROOT / "scripts/build_sft_v2_dataset.py", ROOT / "scripts/audit_sft_v2_dataset.py", ROOT / "scripts/verify_sft_v2_functional.py", ROOT / "scripts/analyze_sft_v2_sequence.py", ROOT / "scripts/analyze_sft_v2_budget.py", ROOT / "scripts/build_sft_v2_token_cache.py", ROOT / "scripts/preflight_sft_v2.py", ROOT / "scripts/train_sft_v2.py"])
    sources = sorted({path for path in sources if path.is_file()})
    forbidden = [path for path in sources if any(part in {"runs", "checkpoints", "optimizer", "model.pt", "genpy200m_sft_v1"} for part in path.parts) or path.suffix in {".tar", ".pth"}]
    if forbidden:
        raise RuntimeError("package contains forbidden model/training artifacts: " + ",".join(str(path) for path in forbidden))
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT).as_posix())
    listing = {"format_version": 2, "archive": str(output), "sha256": digest(output), "file_count": len(sources), "files": [path.relative_to(ROOT).as_posix() for path in sources], "excludes": ["Checkpoint 7 model.pt", "model archives", "optimizer states", "v1 SFT artifacts", "production SFT outputs"]}
    listing_path = output.with_suffix(".manifest.json"); listing_path.write_text(json.dumps(listing, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(f"{listing['sha256']}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(output), "sha256": listing["sha256"], "file_count": len(sources)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
