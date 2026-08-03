"""Validate generated shards, schemas, checksums, splits, and exact leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpy.data.schemas import PretrainingRecord  # noqa: E402
from genpy.data.sharding import iter_shard_records  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(config_path: Path) -> dict[str, Any]:
    """Return validation counts and raise on any integrity or leakage failure."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_root = ROOT / config["paths"]["splits"]
    database = ROOT / config["paths"]["work"] / "validation.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE hashes (digest TEXT PRIMARY KEY, split TEXT NOT NULL)")
    counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    checksum_count = 0
    for split in counts:
        paths = sorted((split_root / "pretraining" / split).glob("part-*.jsonl.zst"))
        for path in paths:
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["sha256"] != _sha256(path):
                raise ValueError(f"shard checksum mismatch: {path.name}")
            checksum_count += 1
            for value in iter_shard_records((path,)):
                record = PretrainingRecord.from_dict(value)
                if record.split != split:
                    raise ValueError(f"record split mismatch: {record.record_id}")
                row = connection.execute(
                    "SELECT split FROM hashes WHERE digest = ?", (record.content_sha256,)
                ).fetchone()
                if row and str(row[0]) != split:
                    raise ValueError("exact duplicate crosses split boundaries")
                connection.execute(
                    "INSERT OR IGNORE INTO hashes VALUES (?, ?)", (record.content_sha256, split)
                )
                counts[split] += 1
    connection.commit()
    connection.close()
    return {"passed": True, "split_counts": counts, "validated_shards": checksum_count}


def main() -> int:
    """Run dataset validation and print machine-readable results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
