from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from genpy.tokenizer.config import load_tokenizer_config
from genpy.tokenizer.tokenizer import GenPyTokenizer
from genpy.tokenizer.trainer import train_tokenizer
from genpy.training.packing import load_packing_config, prepare_packed_data
from tests.unit._tokenizer_helpers import write_fixture_workspace


@pytest.fixture(scope="session")
def trained_tokenizer_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train one small local tokenizer shared by wrapper-focused tests."""
    root = tmp_path_factory.mktemp("tokenizer-artifact")
    config_path = write_fixture_workspace(root)
    config = load_tokenizer_config(config_path, root)
    artifact = root / "artifacts/tokenizer/fixture-smoke"
    train_tokenizer(config, mode="smoke", output=artifact)
    return artifact


@pytest.fixture(scope="session")
def phase4_fixture(
    tmp_path_factory: pytest.TempPathFactory, trained_tokenizer_artifact: Path
) -> dict[str, Path]:
    """Build clone-safe tiny model and packed-data configs around the fixture tokenizer."""
    root = Path(__file__).resolve().parents[1]
    workspace = tmp_path_factory.mktemp("phase4")
    tokenizer = GenPyTokenizer.load(trained_tokenizer_artifact)
    model_raw = yaml.safe_load((root / "configs/model/smoke_model.yaml").read_text())
    model_raw["model"].update(
        {
            "vocab_size": tokenizer.vocab_size,
            "context_length": 16,
            "num_layers": 1,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dimension": 8,
            "intermediate_size": 64,
        }
    )
    model_raw["tokenizer"].update(
        {
            "name": "genpy-fixture-smoke",
            "vocab_size": tokenizer.vocab_size,
            "artifact_path": str(trained_tokenizer_artifact),
            "fingerprint": tokenizer.fingerprint,
        }
    )
    model_path = workspace / "smoke_model.yaml"
    model_path.write_text(yaml.safe_dump(model_raw, sort_keys=False), encoding="utf-8")
    packing_raw = yaml.safe_load((root / "configs/data/smoke_packing.yaml").read_text())
    output = workspace / "packed"
    packing_raw["packing"].update(
        {
            "context_length": 16,
            "stored_token_width": 17,
            "output_root": str(output),
            "report_json": str(workspace / "packing.json"),
            "report_markdown": str(workspace / "packing.md"),
        }
    )
    packing_raw["tokenizer"].update(
        {
            "name": "genpy-fixture-smoke",
            "vocab_size": tokenizer.vocab_size,
            "artifact_path": str(trained_tokenizer_artifact),
            "fingerprint": tokenizer.fingerprint,
        }
    )
    packing_raw["source"]["fixture_path"] = str(
        root / "tests/fixtures/packed_data/smoke_records.json"
    )
    packing_path = workspace / "smoke_packing.yaml"
    packing_path.write_text(yaml.safe_dump(packing_raw, sort_keys=False), encoding="utf-8")
    packing = load_packing_config(packing_path, root)
    prepare_packed_data(packing)
    return {
        "root": root,
        "workspace": workspace,
        "model_config": model_path,
        "packing_config": packing_path,
        "packed_root": output,
        "tokenizer_artifact": trained_tokenizer_artifact,
    }
