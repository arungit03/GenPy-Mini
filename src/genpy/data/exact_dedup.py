"""SQLite-backed global exact deduplication."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from genpy.data.schemas import PretrainingRecord


def provenance_rank(record: PretrainingRecord) -> float:
    """Rank records so duplicate winners have stronger provenance and quality."""
    return (
        record.quality.quality_score
        + (0.05 if record.repository else 0.0)
        + (0.05 if record.source_url.startswith("https://") else 0.0)
        + (0.05 if record.licence_spdx else 0.0)
    )


class ExactDeduplicator:
    """Retain one best record per content hash without an in-memory corpus index."""

    def __init__(self, database_path: Path, duplicate_manifest: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        duplicate_manifest.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS winners ("
            "content_hash TEXT PRIMARY KEY, record_id TEXT NOT NULL, rank REAL NOT NULL, "
            "record_json TEXT NOT NULL)"
        )
        self._manifest = duplicate_manifest
        self.duplicates = 0
        self._pending_commits = 0

    def add(self, record: PretrainingRecord) -> bool:
        """Add a record and return true if it is currently the exact-hash winner."""
        self._pending_commits += 1
        if self._pending_commits % 500 == 0:
            self._connection.commit()
        row = self._connection.execute(
            "SELECT record_id, rank FROM winners WHERE content_hash = ?",
            (record.content_sha256,),
        ).fetchone()
        rank = provenance_rank(record)
        if row is None:
            self._connection.execute(
                "INSERT INTO winners VALUES (?, ?, ?, ?)",
                (record.content_sha256, record.record_id, rank, json.dumps(record.to_dict())),
            )
            return True
        self.duplicates += 1
        previous_id, previous_rank = str(row[0]), float(row[1])
        replaced = rank > previous_rank or (
            rank == previous_rank and record.record_id < previous_id
        )
        removed_id = previous_id if replaced else record.record_id
        kept_id = record.record_id if replaced else previous_id
        if replaced:
            self._connection.execute(
                "UPDATE winners SET record_id = ?, rank = ?, record_json = ? "
                "WHERE content_hash = ?",
                (record.record_id, rank, json.dumps(record.to_dict()), record.content_sha256),
            )
        with self._manifest.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "reason": "exact_duplicate",
                        "content_sha256": record.content_sha256,
                        "kept_record_id": kept_id,
                        "removed_record_id": removed_id,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        return replaced

    def iter_winners(self) -> Iterator[PretrainingRecord]:
        """Yield winners in strongest-provenance order for near deduplication."""
        self._connection.commit()
        cursor = self._connection.execute(
            "SELECT record_json FROM winners ORDER BY rank DESC, record_id ASC"
        )
        for (payload,) in cursor:
            yield PretrainingRecord.from_dict(json.loads(str(payload)))

    def close(self) -> None:
        """Commit and close the on-disk index."""
        self._connection.commit()
        self._connection.close()


def deduplicate_exact(records: list[PretrainingRecord], work_dir: Path) -> list[PretrainingRecord]:
    """Convenience helper for bounded tests and small tools."""
    deduplicator = ExactDeduplicator(work_dir / "exact.sqlite3", work_dir / "exact.jsonl")
    try:
        for record in records:
            deduplicator.add(record)
        return list(deduplicator.iter_winners())
    finally:
        deduplicator.close()
