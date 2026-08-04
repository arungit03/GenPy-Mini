from __future__ import annotations

import shutil

import pytest

from genpy.training.packed_format import validate_packed_manifest
from genpy.training.packing import load_packing_config


def test_packed_checksums_and_corruption_detection(tmp_path, phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    config = load_packing_config(phase4_fixture["packing_config"], phase4_fixture["root"])
    copied = tmp_path / "packed"
    shutil.copytree(config.output_root, copied)
    manifest = copied / "manifests/packing_manifest.json"
    result = validate_packed_manifest(
        manifest, str(config.tokenizer["fingerprint"]), config.config_hash
    )
    assert result["passed"] and not result["split_contamination_detected"]
    binary = next(copied.rglob("*.tokens.bin"))
    with binary.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        validate_packed_manifest(
            manifest, str(config.tokenizer["fingerprint"]), config.config_hash
        )
