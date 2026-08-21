import pytest

from genpy.training.config import load_training_config


def test_training_configs_validate() -> None:
    assert load_training_config("configs/training_engine.yaml").training.gradient_accumulation_steps == 8
    assert load_training_config("configs/training_smoke.yaml").training.max_steps == 4


def test_training_budget_and_precision_validation() -> None:
    config = load_training_config("configs/training_engine.yaml")
    with pytest.raises(ValueError, match="without"):
        config.validate(require_budget=True)
