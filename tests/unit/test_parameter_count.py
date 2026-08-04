from __future__ import annotations

from pathlib import Path

from genpy.model.config import load_model_config
from genpy.model.memory import estimate_training_memory
from genpy.model.parameter_count import count_parameters
from genpy.model.transformer import build_model


def test_exact_production_parameter_counts() -> None:
    expected = {
        "genpy_5m.yaml": 4_916_928,
        "genpy_25m.yaml": 20_453_760,
        "genpy_100m.yaml": 97_536_768,
    }
    for name, count in expected.items():
        audit = count_parameters(load_model_config(Path("configs/model") / name))
        assert audit.total_parameters == count


def test_formula_matches_unique_smoke_model_parameters(phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    config = load_model_config(phase4_fixture["model_config"], phase4_fixture["root"])
    model = build_model(config)
    unique = {parameter.data_ptr(): parameter.numel() for parameter in model.parameters()}
    assert sum(unique.values()) == count_parameters(config).total_parameters
    assert model.token_embedding.weight is model.lm_head.weight


def test_memory_estimator_is_allocation_free_and_ordered() -> None:
    config = load_model_config(Path("configs/model/genpy_100m.yaml"))
    report = estimate_training_memory(config, 1024, 1)
    assert report["dtypes"]["float32"]["parameter_bytes"] == 97_536_768 * 4
    assert report["dtypes"]["float32"]["estimated_total_bytes"] > report["dtypes"][
        "float32"
    ]["parameter_bytes"]
