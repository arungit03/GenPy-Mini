"""Package the latest complete resumable SFT checkpoint without dataset files."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-config", default="configs/model_200m.yaml")
    parser.add_argument("--train-config", default="configs/sft_200m_kaggle.yaml")
    parser.add_argument("--data", default="data/instruction/tokenized/SFT_TOKEN_CACHE_MANIFEST.json")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    pointer = run_dir / "checkpoints/latest.json"
    checkpoint_name = json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
    checkpoint = run_dir / "checkpoints" / checkpoint_name
    if not (checkpoint / "COMPLETE").is_file():
        raise ValueError("latest SFT checkpoint is incomplete")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    files = [(pointer, "checkpoints/latest.json"), (Path(args.model_config), "model_config.yaml"), (Path(args.train_config), "training_config.yaml"), (Path(args.data), "sft_token_cache_manifest.json"), (run_dir / "run_manifest.json", "run_manifest.json"), (run_dir / "logs/training_metrics.jsonl", "training_metrics.jsonl")]
    files.extend((source, str(Path("checkpoints") / checkpoint_name / source.relative_to(checkpoint))) for source in checkpoint.rglob("*") if source.is_file())
    if any(not source.is_file() for source, _ in files):
        raise FileNotFoundError("SFT package metadata is incomplete")
    with tarfile.open(output, "w") as archive:
        for source, name in files:
            archive.add(source, arcname=name, recursive=False)
    checksum = digest(output)
    output.with_name(output.name + ".sha256").write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    print(f"Packaged SFT checkpoint {checkpoint_name}: {output}\nSHA256: {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
