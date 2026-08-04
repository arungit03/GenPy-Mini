from __future__ import annotations

import shutil

import pytest

from genpy.tokenizer.tokenizer import GenPyTokenizer, TokenizerArtifactError


def test_invalid_token_ids_are_rejected(trained_tokenizer_artifact) -> None:  # type: ignore[no-untyped-def]
    tokenizer = GenPyTokenizer.load(trained_tokenizer_artifact)
    with pytest.raises(ValueError):
        tokenizer.decode([-1])
    with pytest.raises(ValueError):
        tokenizer.decode([tokenizer.vocab_size])


def test_missing_file_and_corrupted_checksum_fail(trained_tokenizer_artifact, tmp_path) -> None:  # type: ignore[no-untyped-def]
    missing = tmp_path / "missing"
    shutil.copytree(trained_tokenizer_artifact, missing)
    (missing / "merges.txt").unlink()
    with pytest.raises(TokenizerArtifactError, match="missing"):
        GenPyTokenizer.load(missing)

    corrupted = tmp_path / "corrupted"
    shutil.copytree(trained_tokenizer_artifact, corrupted)
    (corrupted / "vocab.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TokenizerArtifactError, match="checksum"):
        GenPyTokenizer.load(corrupted)
