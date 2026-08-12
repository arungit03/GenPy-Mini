import torch

from genpy.model import GenPyForCausalLM
from genpy.training.optimizer import create_adamw, parameter_groups
from genpy.config import TrainingConfig
from tests.test_model_architecture import tiny_config


def train_config():
    return TrainingConfig(1, 2, 1, 1e-3, 1e-4, 0.1, 0.1, 1.0, "fp32", 1, 2, 3, "checkpoints", "logs")


def test_optimizer_groups_are_unique_and_decay_only_matrices():
    model = GenPyForCausalLM(tiny_config())
    groups = parameter_groups(model, 0.1)
    all_parameters = [parameter for group in groups for parameter in group["params"]]
    assert len(all_parameters) == len({id(parameter) for parameter in all_parameters})
    assert {id(parameter) for parameter in all_parameters} == {id(parameter) for parameter in model.parameters()}
    assert all(parameter.ndim >= 2 for parameter in groups[0]["params"])
    assert all(parameter.ndim < 2 for parameter in groups[1]["params"])
    optimizer = create_adamw(model, train_config())
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert optimizer.defaults["eps"] == 1e-8
