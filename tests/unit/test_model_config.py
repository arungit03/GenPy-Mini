from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from genpy.model.config import ModelConfigError, load_model_config
from genpy.model.readiness import check_model_readiness


def test_production_profiles_preserve_phase1_dimensions() -> None:
    expected = {
        "genpy_5m.yaml": (4, 192, 4, 512),
        "genpy_25m.yaml": (8, 384, 6, 1024),
        "genpy_100m.yaml": (12, 768, 12, 2048),
    }
    for name, dimensions in expected.items():
        config = load_model_config(Path("configs/model") / name)
        assert (
            config.num_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
        ) == dimensions
        assert config.vocab_size == 16384
        assert config.context_length == 1024


def test_unknown_and_inconsistent_config_values_fail(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/model/smoke_model.yaml").read_text())
    raw["model"]["unknown"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ModelConfigError, match="unknown"):
        load_model_config(path)


def test_production_readiness_does_not_claim_phase5() -> None:
    result = check_model_readiness(Path("configs/model/genpy_5m.yaml"))
    assert result.status in {"READY_FOR_SMOKE_MODEL", "NOT_READY"}
    checks = {check.name: check.passed for check in result.checks}
    assert not checks["tokenizer_contract_matches"]
    assert not checks["all_production_splits_packed"]
