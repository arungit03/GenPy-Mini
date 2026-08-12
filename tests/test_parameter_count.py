from pathlib import Path

from genpy.config import load_model_config
from genpy.model import GenPyForCausalLM


ROOT = Path(__file__).resolve().parents[1]


def test_production_parameter_count_and_breakdown():
    config = load_model_config(ROOT / "configs" / "model_200m.yaml")
    model = GenPyForCausalLM(config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    assert trainable == 201_560_832
    assert total == trainable
    breakdown = model.parameter_breakdown()
    assert sum(breakdown.values()) == trainable
    assert breakdown["Embedding"] == 32_000 * 768
    assert breakdown["Attention"] == 24 * 4 * 768 * 768
    assert breakdown["SwiGLU"] == 24 * 3 * 768 * 2176
    assert breakdown["RMSNorm"] == 24 * 2 * 768 + 768
