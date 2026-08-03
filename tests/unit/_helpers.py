"""Original test-only record builders."""

from __future__ import annotations

from genpy.data.schemas import PretrainingRecord, QualityInfo, content_sha256, stable_record_id


def make_record(
    text: str,
    *,
    identity: str = "example.py",
    group: str = "example/project",
    score: float = 0.9,
) -> PretrainingRecord:
    """Build a valid fictional pretraining record."""
    digest = content_sha256(text)
    return PretrainingRecord(
        record_id=stable_record_id("test-fixture", identity, digest),
        text=text,
        language="python",
        source_id="test-fixture",
        source_url="https://example.invalid/source",
        repository=group,
        revision="fixture-v1",
        file_path=identity,
        licence_spdx="MIT",
        content_sha256=digest,
        generation_method="human",
        quality=QualityInfo(True, True, True, score),
        split_group=group,
    )
