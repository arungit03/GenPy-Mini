from dataclasses import replace
from pathlib import Path

import pytest

from genpy.config import DataPipelineConfig, load_data_config


ROOT = Path(__file__).resolve().parents[1]


def test_load_data_configuration():
    config = load_data_config(ROOT / "configs" / "data.yaml")
    assert config.dataset.name == "HuggingFaceFW/fineweb-edu"
    assert config.dataset.config == "sample-10BT"
    assert config.dataset.streaming is True
    assert config.processing.min_chars == 200
    assert config.output.format == "jsonl.gz"


def test_data_configuration_requires_sections():
    with pytest.raises(ValueError, match="processing"):
        DataPipelineConfig.from_mapping({"dataset": {}})


def _mapping():
    return {
        "dataset": {"name": "x", "config": "y", "split": "train", "streaming": True, "text_field": "text"},
        "processing": {"seed": 0, "min_chars": 2, "max_chars": 10, "normalize_unicode": True, "normalize_line_endings": True, "remove_control_characters": True, "normalize_whitespace": True, "exact_deduplication": True},
        "split": {"validation_fraction": 0.2, "seed": 0},
        "output": {"format": "jsonl.gz", "shard_max_documents": 2, "processed_dir": "data/processed", "manifest_dir": "data/manifests"},
        "resume": {"enabled": True},
        "metadata": {"preserve_source_metadata": True},
    }


@pytest.mark.parametrize("change", [
    {"min_chars": 0}, {"max_chars": 2},
])
def test_invalid_processing_thresholds(change):
    mapping = _mapping()
    mapping["processing"].update(change)
    with pytest.raises(ValueError):
        DataPipelineConfig.from_mapping(mapping)


def test_invalid_split_and_output():
    mapping = _mapping()
    mapping["split"]["validation_fraction"] = 1
    with pytest.raises(ValueError, match="validation_fraction"):
        DataPipelineConfig.from_mapping(mapping)
    mapping = _mapping()
    mapping["output"]["format"] = "jsonl"
    with pytest.raises(ValueError, match="jsonl.gz"):
        DataPipelineConfig.from_mapping(mapping)
