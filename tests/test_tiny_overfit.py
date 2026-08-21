from genpy.config import load_config
from genpy.verification.overfit import run_tiny_overfit


def test_tiny_model_memorizes_fixed_batch() -> None:
    result = run_tiny_overfit(load_config("configs/model_200m.yaml").model, steps=200)
    assert result["passed"]
    assert result["final_loss"] < result["initial_loss"]
