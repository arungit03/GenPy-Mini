import json
from pathlib import Path

import pytest

from genpy.config import load_tokenizer_config
from genpy.tokenizer.corpus import CorpusReader
from genpy.tokenizer.trainer import save_tokenizer_artifacts, train_from_iterator
from genpy.tokenizer.validation import TokenizerValidationError, validate_tokenizer_artifact


ROOT = Path(__file__).resolve().parents[1]


def _artifact(tmp_path):
    config = load_tokenizer_config(ROOT / "configs" / "tokenizer.yaml")
    reader = CorpusReader(ROOT / "tests" / "fixtures", "tokenizer_corpus.jsonl.gz")
    tokenizer = train_from_iterator(config.tokenizer, reader.texts(max_documents=120), vocab_size=512)
    smoke_config = config
    result = save_tokenizer_artifacts(tokenizer, smoke_config, tmp_path, reader.stats, "smoke")
    return result


def test_validation_and_checksum(tmp_path):
    result = _artifact(tmp_path)
    report = validate_tokenizer_artifact(result["tokenizer_path"], result["manifest_path"], expected_vocab_size=512)
    assert report["round_trip"]


def test_validation_detects_bad_checksum(tmp_path):
    result = _artifact(tmp_path)
    data = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    data["tokenizer_json_sha256"] = "bad"
    result["manifest_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TokenizerValidationError, match="checksum"):
        validate_tokenizer_artifact(result["tokenizer_path"], result["manifest_path"], expected_vocab_size=512)


def test_validation_detects_wrong_vocab(tmp_path):
    result = _artifact(tmp_path)
    with pytest.raises(TokenizerValidationError, match="Vocabulary"):
        validate_tokenizer_artifact(result["tokenizer_path"], expected_vocab_size=32000)
