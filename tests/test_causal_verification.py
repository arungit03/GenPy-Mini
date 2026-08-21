import torch

from _verification_cli import tiny_config
from genpy.model import GenPyForCausalLM
from genpy.verification.causal import causal_isolation, intermediate_layer_isolation


def test_deep_causal_isolation() -> None:
    model = GenPyForCausalLM(tiny_config(), attention_backend="eager").eval()
    first = torch.tensor([[10, 20, 30, 40, 50]])
    second = torch.tensor([[10, 20, 30, 99, 88]])
    assert causal_isolation(model, first, second, 3)["passed"]
    assert intermediate_layer_isolation(model, first, second, 3)["passed"]
