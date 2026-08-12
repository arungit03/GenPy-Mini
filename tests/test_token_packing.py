import json

import numpy as np

from genpy.training.data import PackedTokenDataset


def test_packing_window_boundaries(tmp_path):
    values = np.arange(13, dtype=np.uint16)
    values.tofile(tmp_path / "validation.bin")
    (tmp_path / "validation_metadata.json").write_text(json.dumps({"dtype": "uint16", "token_count": 13, "vocab_size": 20}), encoding="utf-8")
    dataset = PackedTokenDataset(tmp_path / "validation.bin", 4)
    assert len(dataset) == 3
    for index in range(3):
        assert dataset[index]["targets"][0].item() == dataset[index]["input_ids"][0].item() + 1
