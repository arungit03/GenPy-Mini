from genpy.data.dedup import ExactDeduplicator, content_hash


def test_sha256_exact_deduplication():
    digest = content_hash("same")
    assert len(digest) == 64
    dedup = ExactDeduplicator()
    assert dedup.add(digest)
    assert not dedup.add(digest)
    assert dedup.accept("different")
    assert content_hash("same") != content_hash("different")
