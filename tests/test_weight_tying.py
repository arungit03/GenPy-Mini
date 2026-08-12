import io

import pytest
import torch

from tests.test_model_architecture import tiny_config
from genpy.model import GenPyForCausalLM


def test_embedding_and_lm_head_are_the_same_parameter():
    model = GenPyForCausalLM(tiny_config())
    assert model.lm_head.weight is model.token_embedding.weight
    before = model.lm_head.weight[0, 0].item()
    with torch.no_grad():
        model.token_embedding.weight[0, 0].add_(1.0)
    assert model.lm_head.weight[0, 0].item() == pytest.approx(before + 1.0)
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    assert parameter_ids.count(id(model.token_embedding.weight)) == 1


def test_state_dict_round_trip_preserves_tying_and_logits():
    torch.manual_seed(7)
    first = GenPyForCausalLM(tiny_config()).eval()
    input_ids = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        expected = first(input_ids)
    buffer = io.BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    second = GenPyForCausalLM(tiny_config()).eval()
    second.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        actual = second(input_ids)
    assert second.lm_head.weight is second.token_embedding.weight
    assert torch.equal(expected, actual)
