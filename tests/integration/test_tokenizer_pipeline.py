from __future__ import annotations

import json
from pathlib import Path

import yaml

from genpy.tokenizer.config import SPECIAL_TOKEN_TEXT, load_tokenizer_config
from genpy.tokenizer.corpus import prepare_corpus_manifest
from genpy.tokenizer.evaluation import (
    count_corpus_tokens,
    evaluate_tokenizer,
    package_artifact,
    vocabulary_audit,
)
from genpy.tokenizer.tokenizer import GenPyTokenizer
from genpy.tokenizer.trainer import train_tokenizer
from genpy.tokenizer.validation import check_readiness
from tests.unit._tokenizer_helpers import write_fixture_workspace


def test_end_to_end_deterministic_smoke_tokenizer(tmp_path: Path) -> None:
    config_path = write_fixture_workspace(tmp_path)
    config = load_tokenizer_config(config_path, tmp_path)
    readiness = check_readiness(config)
    assert readiness.status == "READY_FOR_SMOKE_TOKENIZER"
    summary = prepare_corpus_manifest(config)
    assert summary["selected_records"] == 3
    assert summary["actual_mixture"] == {"pretraining": 1.0}

    first_artifact = tmp_path / "artifacts/tokenizer/run-one"
    second_artifact = tmp_path / "artifacts/tokenizer/run-two"
    first_metadata = train_tokenizer(config, mode="smoke", output=first_artifact)
    second_metadata = train_tokenizer(config, mode="smoke", output=second_artifact)
    assert first_metadata["actual_vocabulary_size"] == 500
    assert first_metadata["tokenizer_fingerprint"] == second_metadata["tokenizer_fingerprint"]
    assert (first_artifact / "vocab.json").read_bytes() == (
        second_artifact / "vocab.json"
    ).read_bytes()
    assert (first_artifact / "merges.txt").read_bytes() == (
        second_artifact / "merges.txt"
    ).read_bytes()

    tokenizer = GenPyTokenizer.load(first_artifact)
    sample = "def greet(name):\n    return f'Hello {name} 😀'\n"
    assert tokenizer.decode(tokenizer.encode_text(sample)) == sample
    for expected_id, token in enumerate(SPECIAL_TOKEN_TEXT):
        assert tokenizer.token_to_id(token) == expected_id
    assert package_artifact(first_artifact)
    assert GenPyTokenizer.load(first_artifact).fingerprint == tokenizer.fingerprint

    audit = vocabulary_audit(tokenizer, first_artifact, maximum_token_length=128)
    assert audit["structural_errors"] == []
    counts = count_corpus_tokens(config, first_artifact, resume=False)
    assert counts["splits"]["pretraining_train"]["record_count"] == 3
    assert counts["splits"]["pretraining_validation"]["record_count"] == 0

    evaluation_config = {
        "schema_version": 1,
        "tokenizer_config": str(config_path),
        "artifact_path": str(first_artifact.relative_to(tmp_path)),
        "report_json": "data/tokenizer/reports/evaluation.json",
        "report_markdown": "data/tokenizer/reports/evaluation.md",
        "training_sample_records": 3,
        "context_length": 1024,
        "security": {"maximum_token_length": 128},
    }
    evaluation_path = tmp_path / "configs/tokenizer/evaluation.yaml"
    evaluation_path.write_text(yaml.safe_dump(evaluation_config), encoding="utf-8")
    evaluation = evaluate_tokenizer(evaluation_path)
    assert evaluation["evaluation_sets"]["deterministic_training_sample"]["record_count"] == 3
    assert evaluation["evaluation_sets"]["phase2_validation"]["record_count"] == 0
    assert json.loads((first_artifact / "metadata.json").read_text())["status"] == "smoke"
