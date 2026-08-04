from __future__ import annotations

import json

import pytest

from genpy.tokenizer.config import load_tokenizer_config
from genpy.tokenizer.corpus import CorpusError, iter_family_items, prepare_corpus_manifest
from genpy.tokenizer.validation import check_readiness
from tests.unit._tokenizer_helpers import write_fixture_workspace


def test_stable_training_only_manifest_has_no_text(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = load_tokenizer_config(write_fixture_workspace(tmp_path), tmp_path)
    first = prepare_corpus_manifest(config)
    manifest = config.resolve(config.corpus["manifest_path"])
    first_bytes = manifest.read_bytes()
    second = prepare_corpus_manifest(config)
    assert first["corpus_fingerprint"] == second["corpus_fingerprint"]
    assert manifest.read_bytes() == first_bytes
    for line in manifest.read_text().splitlines():
        value = json.loads(line)
        assert value["split"] == "train"
        assert "text" not in value


def test_validation_or_test_contamination_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = load_tokenizer_config(
        write_fixture_workspace(tmp_path, contaminated_split="validation"), tmp_path
    )
    with pytest.raises(CorpusError, match="non-training"):
        list(iter_family_items(config, "pretraining"))
    assert check_readiness(config).status == "NOT_READY"


def test_failed_phase2_safety_flag_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = load_tokenizer_config(
        write_fixture_workspace(tmp_path, unsafe_quality=True), tmp_path
    )
    with pytest.raises(CorpusError, match="unapproved safety flags"):
        list(iter_family_items(config, "pretraining"))
    assert check_readiness(config).status == "NOT_READY"
