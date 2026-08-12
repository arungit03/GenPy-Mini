import gzip
import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from genpy.tokenizer.tokenizer import GenPyTokenizer
from genpy.training.tokenization import prepare_tokenized_split


def production_contract_test_tokenizer(path):
    vocab = {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2, "<|unk|>": 3, "hello": 4, "world": 5}
    vocab.update({f"<extra_{index}>": index + 6 for index in range(31994)})
    raw = Tokenizer(WordLevel(vocab=vocab, unk_token="<|unk|>"))
    wrapper = GenPyTokenizer(raw)
    wrapper.save(path)


def test_tokenized_preparation_requires_a_valid_production_tokenizer(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    with gzip.open(shard_dir / "train_00000.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "hello"}) + "\n")
    with pytest.raises(FileNotFoundError):
        prepare_tokenized_split(shard_dir, tmp_path / "missing-tokenizer.json", tmp_path / "tokens", "train")


def test_streaming_token_preparation_writes_eos_and_metadata(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    with gzip.open(shard_dir / "train_00000.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "hello world"}) + "\n")
        handle.write(json.dumps({"text": "hello"}) + "\n")
    tokenizer_path = tmp_path / "tokenizer.json"
    production_contract_test_tokenizer(tokenizer_path)
    metadata = prepare_tokenized_split(shard_dir, tokenizer_path, tmp_path / "tokens", "train")
    assert metadata["dtype"] == "uint16"
    assert metadata["document_count"] == 2
    assert metadata["eos_count"] == 2
    assert (tmp_path / "tokens" / "train.bin").is_file()
    assert (tmp_path / "tokens" / "train_metadata.json").is_file()
