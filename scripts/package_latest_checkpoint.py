"""Package only the latest resumable checkpoint and its provenance files."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-config", default="configs/model_200m.yaml")
    parser.add_argument("--train-config", default="configs/pretrain_200m_kaggle.yaml")
    parser.add_argument("--data", default="data/tokenized/genpy-32k/TOKEN_CACHE_MANIFEST.json")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    pointer = run_dir / "checkpoints/latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"missing checkpoint pointer: {pointer}")
    checkpoint_name = json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
    checkpoint = run_dir / "checkpoints" / checkpoint_name
    if not (checkpoint / "COMPLETE").is_file():
        raise ValueError("latest checkpoint is incomplete")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files: list[tuple[Path, str]] = [(pointer, "checkpoints/latest.json"), (Path(args.model_config), "model_config.yaml"), (Path(args.train_config), "training_config.yaml"), (Path(args.data), "token_cache_manifest.json"), (run_dir / "run_manifest.json", "run_manifest.json"), (run_dir / "logs/training_metrics.jsonl", "training_metrics.jsonl"), (checkpoint / "metadata.json", "checkpoint_metadata.json")]
    files.extend((path, str(Path("checkpoints") / checkpoint_name / path.relative_to(checkpoint))) for path in checkpoint.rglob("*") if path.is_file())
    missing = [str(source) for source, _ in files if not source.is_file()]
    if missing:
        raise FileNotFoundError("required package files missing: " + ", ".join(missing))
    with tarfile.open(output, mode="w") as archive:
        for source, name in files:
            archive.add(source, arcname=name, recursive=False)
    checksum = sha256(output)
    checksum_path = output.with_name(output.name + ".sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    names = [name for _, name in files]
    assert not any(name.endswith(("train.bin", "validation.bin")) for name in names)
    print(f"Packaged {checkpoint_name}: {output}")
    print(f"SHA256: {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
