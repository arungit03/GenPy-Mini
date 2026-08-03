from __future__ import annotations

import hashlib
import json

from genpy.data.sharding import iter_shard_records, write_shards
from tests.unit._helpers import make_record


def test_shard_checksum_and_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    record = make_record("answer = 42\n")
    manifests = write_shards(
        [record.to_dict()],
        tmp_path,
        maximum_uncompressed_bytes=1024,
        compression_level=1,
        config_hash="fixture-config",
    )
    shard = tmp_path / manifests[0]["shard"]
    assert hashlib.sha256(shard.read_bytes()).hexdigest() == manifests[0]["sha256"]
    values = list(iter_shard_records((shard,)))
    assert values[0]["record_id"] == record.record_id
    sidecar = json.loads(shard.with_suffix(shard.suffix + ".manifest.json").read_text())
    assert sidecar["record_count"] == 1
