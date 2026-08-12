"""Deterministic content-hash-based train/validation assignment."""

import hashlib


def assign_split(content_hash_value: str, validation_fraction: float = 0.005, seed: int = 42) -> str:
    """Assign a stable split using the content hash and split seed."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    material = f"{content_hash_value}:{seed}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(material).digest(), "big") / float(2 ** 256)
    return "validation" if bucket < validation_fraction else "train"
