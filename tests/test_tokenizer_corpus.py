import gzip
import json

import pytest

from genpy.tokenizer.corpus import CorpusReader


def _write(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_corpus_reader_order_limits_and_stats(tmp_path):
    _write(tmp_path / "train-00001.jsonl.gz", [{"text": "second"}])
    _write(tmp_path / "train-00000.jsonl.gz", [{"text": "first"}, {"text": "third"}])
    reader = CorpusReader(tmp_path, "train-*.jsonl.gz")
    assert list(reader.texts(max_documents=2)) == ["first", "third"]
    assert reader.stats.documents == 2
    assert reader.stats.utf8_bytes == len("firstthird".encode("utf-8"))


def test_corpus_reader_validation_selection_and_missing_text(tmp_path):
    _write(tmp_path / "validation-00000.jsonl.gz", [{"text": "validation"}])
    assert list(CorpusReader(tmp_path, "validation-*.jsonl.gz").texts()) == ["validation"]
    _write(tmp_path / "train-00000.jsonl.gz", [{"other": "missing"}])
    with pytest.raises(ValueError, match="text"):
        list(CorpusReader(tmp_path, "train-*.jsonl.gz").texts())


def test_corpus_reader_malformed_json(tmp_path):
    with gzip.open(tmp_path / "train-00000.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("not json\n")
    with pytest.raises(ValueError, match="Malformed JSON"):
        list(CorpusReader(tmp_path, "train-*.jsonl.gz").rows())
