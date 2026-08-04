from __future__ import annotations

from genpy.tokenizer.config import load_tokenizer_config
from genpy.tokenizer.corpus import prepare_corpus_manifest
from genpy.tokenizer.tokenizer import GenPyTokenizer, TokenizerArtifactError
from tests.unit._tokenizer_helpers import write_fixture_workspace


def test_corpus_fingerprint_is_stable_and_changes_with_input(tmp_path) -> None:  # type: ignore[no-untyped-def]
    first_root = tmp_path / "first"
    first_config = load_tokenizer_config(write_fixture_workspace(first_root), first_root)
    first = prepare_corpus_manifest(first_config)
    repeated = prepare_corpus_manifest(first_config)
    assert first["corpus_fingerprint"] == repeated["corpus_fingerprint"]

    changed_root = tmp_path / "changed"
    changed_config = load_tokenizer_config(
        write_fixture_workspace(changed_root, marker="changed"), changed_root
    )
    changed = prepare_corpus_manifest(changed_config)
    assert changed["corpus_fingerprint"] != first["corpus_fingerprint"]


def test_incorrect_metadata_fingerprint_is_rejected(
    trained_tokenizer_artifact, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    import json
    import shutil

    from genpy.tokenizer.evaluation import package_artifact

    artifact = tmp_path / "fingerprint"
    shutil.copytree(trained_tokenizer_artifact, artifact)
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["tokenizer_fingerprint"] = "incorrect"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    package_artifact(artifact)
    try:
        GenPyTokenizer.load(artifact)
    except TokenizerArtifactError:
        return
    raise AssertionError("invalid tokenizer fingerprint was accepted")
