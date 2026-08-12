import torch

from genpy.model import GenPyForCausalLM
from tests.test_model_architecture import tiny_config


def test_future_token_cannot_change_earlier_model_outputs():
    torch.manual_seed(13)
    model = GenPyForCausalLM(tiny_config()).eval()
    prefix = torch.randint(0, 256, (1, 4))
    first = torch.cat((prefix, torch.tensor([[5]])), dim=1)
    second = torch.cat((prefix, torch.tensor([[17]])), dim=1)
    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)
    assert torch.allclose(first_logits[:, :4], second_logits[:, :4], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(first_logits[:, 4], second_logits[:, 4])
