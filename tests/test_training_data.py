import json

import numpy as np
import pytest

from genpy.training.data import PackedTokenDataset


def write_tokens(tmp_path, values, metadata=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "train.bin"
    np.asarray(values, dtype=np.uint16).tofile(path)
    data = {"format_version": 1, "dtype": "uint16", "split": "train", "token_count": len(values), "document_count": 1, "vocab_size": 100, "special_token_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3}}
    if metadata:
        data.update(metadata)
    (tmp_path / "train_metadata.json").write_text(json.dumps(data), encoding="utf-8")
    return path


def test_packed_dataset_uses_mmap_and_validates_metadata(tmp_path):
    dataset = PackedTokenDataset(write_tokens(tmp_path, range(10)), sequence_length=4)
    assert len(dataset) == 2
    assert isinstance(dataset.tokens, np.memmap)
    assert dataset[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert dataset[0]["targets"].tolist() == [1, 2, 3, 4]
    assert dataset[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert dataset[1]["targets"].tolist() == [5, 6, 7, 8]
    assert dataset[0]["input_ids"].dtype == __import__("torch").int64
    bad = write_tokens(tmp_path / "bad", range(4), {"dtype": "float32"})
    with pytest.raises(ValueError):
        PackedTokenDataset(bad, sequence_length=2)


def test_packed_dataset_retains_eos_and_discards_only_tail(tmp_path):
    dataset = PackedTokenDataset(write_tokens(tmp_path, [5, 6, 2, 7, 8, 2, 9]), sequence_length=3)
    assert len(dataset) == 2
    assert 2 in dataset[0]["input_ids"].tolist() or 2 in dataset[0]["targets"].tolist()
