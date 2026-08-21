from genpy.config import load_config
from genpy.model import GenPyForCausalLM
from genpy.model.utils import count_parameters, parameter_breakdown


def test_production_parameter_count_and_breakdown() -> None:
    model = GenPyForCausalLM(load_config("configs/model_200m.yaml").model)
    assert count_parameters(model) == 201_560_832
    assert parameter_breakdown(model) == {
        "embeddings": 24_576_000,
        "attention": 56_623_104,
        "swiglu": 120_324_096,
        "block_norms": 36_864,
        "final_norm": 768,
    }
