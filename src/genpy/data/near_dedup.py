"""Deterministic shingle MinHash/LSH near-duplicate filtering."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from genpy.data.schemas import PretrainingRecord

TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|[^\s]")
MAX_HASH = (1 << 64) - 1


def token_shingles(text: str, size: int) -> set[str]:
    """Return normalized lexical shingles for similarity comparison."""
    tokens = TOKEN.findall(text)
    if len(tokens) < size:
        return {"\x1f".join(tokens)} if tokens else set()
    return {"\x1f".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Compute exact Jaccard similarity for candidate pairs only."""
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _minhash(shingles: set[str], permutations: int) -> tuple[int, ...]:
    if not shingles:
        return tuple(MAX_HASH for _ in range(permutations))
    signature: list[int] = []
    encoded = [shingle.encode("utf-8") for shingle in shingles]
    for permutation in range(permutations):
        salt = permutation.to_bytes(4, "big")
        signature.append(
            min(
                int.from_bytes(hashlib.blake2b(salt + item, digest_size=8).digest(), "big")
                for item in encoded
            )
        )
    return tuple(signature)


def _bucket_keys(signature: tuple[int, ...], bands: int) -> Iterator[str]:
    rows = len(signature) // bands
    if rows < 1 or rows * bands != len(signature):
        raise ValueError("minhash_permutations must be divisible by lsh_bands")
    for band in range(bands):
        values = signature[band * rows : (band + 1) * rows]
        payload = b"".join(value.to_bytes(8, "big") for value in values)
        yield f"{band}:{hashlib.sha256(payload).hexdigest()[:24]}"


def minhash_bucket_keys(
    shingles: set[str], *, permutations: int = 64, bands: int = 16
) -> tuple[str, ...]:
    """Return deterministic LSH bucket keys for a shingle set."""
    return tuple(_bucket_keys(_minhash(shingles, permutations), bands))


class NearDeduplicator:
    """Use on-disk LSH buckets to avoid all-pairs corpus comparison."""

    def __init__(
        self,
        database_path: Path,
        duplicate_manifest: Path,
        *,
        threshold: float = 0.85,
        shingle_size: int = 5,
        permutations: int = 64,
        bands: int = 16,
    ) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        duplicate_manifest.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS records "
            "(record_id TEXT PRIMARY KEY, record_json TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS buckets (bucket TEXT NOT NULL, record_id TEXT NOT NULL, "
            "PRIMARY KEY (bucket, record_id))"
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS bucket_lookup ON buckets(bucket)")
        self._manifest = duplicate_manifest
        self._threshold = threshold
        self._shingle_size = shingle_size
        self._permutations = permutations
        self._bands = bands
        self.duplicates = 0
        self.cluster_sizes: dict[str, int] = {}
        self._pending_commits = 0

    def add(self, record: PretrainingRecord) -> bool:
        """Keep a record only when no LSH candidate exceeds the threshold."""
        self._pending_commits += 1
        if self._pending_commits % 500 == 0:
            self._connection.commit()
        shingles = token_shingles(record.text, self._shingle_size)
        keys = minhash_bucket_keys(
            shingles, permutations=self._permutations, bands=self._bands
        )
        placeholders = ",".join("?" for _ in keys)
        candidates = self._connection.execute(
            f"SELECT DISTINCT r.record_id, r.record_json FROM records r "  # noqa: S608
            f"JOIN buckets b ON b.record_id = r.record_id WHERE b.bucket IN ({placeholders})",
            keys,
        )
        for candidate_id, payload in candidates:
            candidate = PretrainingRecord.from_dict(json.loads(str(payload)))
            similarity = jaccard_similarity(
                shingles, token_shingles(candidate.text, self._shingle_size)
            )
            if similarity >= self._threshold:
                self.duplicates += 1
                root = str(candidate_id)
                self.cluster_sizes[root] = self.cluster_sizes.get(root, 1) + 1
                with self._manifest.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "reason": "near_duplicate",
                                "kept_record_id": root,
                                "removed_record_id": record.record_id,
                                "similarity": round(similarity, 6),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                return False
        self._connection.execute(
            "INSERT INTO records VALUES (?, ?)", (record.record_id, json.dumps(record.to_dict()))
        )
        self._connection.executemany(
            "INSERT INTO buckets VALUES (?, ?)", ((key, record.record_id) for key in keys)
        )
        return True

    def iter_winners(self) -> Iterator[PretrainingRecord]:
        """Yield accepted records in stable ID order."""
        self._connection.commit()
        cursor = self._connection.execute("SELECT record_json FROM records ORDER BY record_id")
        for (payload,) in cursor:
            yield PretrainingRecord.from_dict(json.loads(str(payload)))

    def close(self) -> None:
        """Commit and close the on-disk LSH index."""
        self._connection.commit()
        self._connection.close()


def deduplicate_near(
    records: list[PretrainingRecord],
    work_dir: Path,
    *,
    threshold: float = 0.85,
) -> list[PretrainingRecord]:
    """Convenience helper for bounded tests and small tools."""
    deduplicator = NearDeduplicator(
        work_dir / "near.sqlite3", work_dir / "near.jsonl", threshold=threshold
    )
    try:
        for record in records:
            deduplicator.add(record)
        return list(deduplicator.iter_winners())
    finally:
        deduplicator.close()
