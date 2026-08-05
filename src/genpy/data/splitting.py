"""Deterministic group-aware splitting and leakage checks."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from genpy.data.near_dedup import jaccard_similarity, minhash_bucket_keys, token_shingles
from genpy.data.schemas import InstructionRecord, PretrainingRecord, SplitName


def choose_split(group: str, ratios: dict[str, float], seed: int) -> SplitName:
    """Map a repository or problem family to a split using a stable hash."""
    required = {"train", "validation", "test"}
    if set(ratios) != required or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must define train/validation/test and sum to 1.0")
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < ratios["train"]:
        return "train"
    if value < ratios["train"] + ratios["validation"]:
        return "validation"
    return "test"


def split_pretraining_records(
    records: Iterable[PretrainingRecord], ratios: dict[str, float], seed: int
) -> Iterator[PretrainingRecord]:
    """Assign all records in one repository/project group to the same split."""
    for record in records:
        yield replace(record, split=choose_split(record.split_group, ratios, seed))


def split_instruction_records(
    records: Iterable[InstructionRecord], ratios: dict[str, float], seed: int
) -> Iterator[InstructionRecord]:
    """Keep all variants of an instruction problem family in one split."""
    for record in records:
        yield replace(record, split=choose_split(record.problem_family, ratios, seed))


def leakage_report(
    records: Iterable[PretrainingRecord], database_path: Path, near_threshold: float = 0.85
) -> dict[str, Any]:
    """Detect exact and candidate near duplicates crossing split boundaries on disk."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS seen (content_hash TEXT, split TEXT, record_id TEXT, text TEXT)"
    )
    connection.execute("CREATE INDEX IF NOT EXISTS seen_hash ON seen(content_hash)")
    connection.execute("CREATE INDEX IF NOT EXISTS seen_record_id ON seen(record_id)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS near_buckets "
        "(bucket TEXT NOT NULL, record_id TEXT NOT NULL, PRIMARY KEY (bucket, record_id))"
    )
    connection.execute("CREATE INDEX IF NOT EXISTS near_bucket_lookup ON near_buckets(bucket)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS groups_seen (split_group TEXT PRIMARY KEY, split TEXT NOT NULL)"
    )
    exact_cross_split = 0
    near_cross_split = 0
    group_violations = 0
    for index, record in enumerate(records):
        row = connection.execute(
            "SELECT split FROM seen WHERE content_hash = ? AND split != ? LIMIT 1",
            (record.content_sha256, record.split),
        ).fetchone()
        if row:
            exact_cross_split += 1
        group_row = connection.execute(
            "SELECT split FROM groups_seen WHERE split_group = ?", (record.split_group,)
        ).fetchone()
        if group_row is None:
            connection.execute(
                "INSERT INTO groups_seen VALUES (?, ?)", (record.split_group, record.split)
            )
        elif str(group_row[0]) != record.split:
            group_violations += 1
        shingles = token_shingles(record.text, 5)
        bucket_keys = minhash_bucket_keys(shingles)
        placeholders = ",".join("?" for _ in bucket_keys)
        candidates = connection.execute(
            f"SELECT DISTINCT s.split, s.text FROM seen s "  # noqa: S608
            f"JOIN near_buckets b ON b.record_id = s.record_id "
            f"WHERE b.bucket IN ({placeholders}) AND s.split != ?",
            (*bucket_keys, record.split),
        )
        for other_split, other_text in candidates:
            if str(other_split) != record.split and jaccard_similarity(
                shingles, token_shingles(str(other_text), 5)
            ) >= near_threshold:
                near_cross_split += 1
                break
        connection.execute(
            "INSERT INTO seen VALUES (?, ?, ?, ?)",
            (record.content_sha256, record.split, record.record_id, record.text),
        )
        connection.executemany(
            "INSERT INTO near_buckets VALUES (?, ?)",
            ((bucket, record.record_id) for bucket in bucket_keys),
        )
        if (index + 1) % 500 == 0:
            connection.commit()
    connection.commit()
    connection.close()
    return {
        "exact_cross_split_duplicates": exact_cross_split,
        "near_cross_split_duplicates": near_cross_split,
        "group_split_violations": group_violations,
        "passed": exact_cross_split == 0 and group_violations == 0,
    }
