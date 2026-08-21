from pathlib import Path

import pytest

from genpy.data.config import load_data_config


def test_dataset_config_loads() -> None:
    config = load_data_config(Path(__file__).parents[1] / "configs" / "data.yaml")
    assert config.target_examples == 100000
    assert config.train_ratio == 0.90
    assert config.deduplication.near_duplicate_threshold == 0.90


def test_dataset_config_rejects_bad_ratios(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("dataset:\n  train_ratio: 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_data_config(path)
