import torch

from genpy.model import GenPyForCausalLM
from genpy.verification import causal_lm_loss, gradient_report
from tests.test_model_architecture import tiny_config


def test_tiny_model_has_finite_gradient_for_every_trainable_parameter():
    torch.manual_seed(12)
    model = GenPyForCausalLM(tiny_config())
    input_ids = torch.randint(0, 256, (2, 16))
    loss = causal_lm_loss(model(input_ids), input_ids)
    assert torch.isfinite(loss)
    loss.backward()
    report = gradient_report(model)
    assert report["checked"] == sum(parameter.requires_grad for parameter in model.parameters())
    assert report["missing"] == []
    assert report["non_finite"] == []
