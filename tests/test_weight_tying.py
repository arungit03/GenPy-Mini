from dataclasses import replace

import torch

from genpy.config import load_config
from genpy.model import GenPyForCausalLM


def test_weight_tying_survives_state_dict_load() -> None:
    config = replace(load_config("configs/model_200m.yaml").model, vocab_size=64, max_seq_len=16, n_layers=1, d_model=16, n_heads=2, head_dim=8, ffn_hidden_size=24)
    first = GenPyForCausalLM(config)
    second = GenPyForCausalLM(config)
    second.load_state_dict(first.state_dict())
    assert first.lm_head.weight is first.token_embedding.weight
    assert second.lm_head.weight is second.token_embedding.weight
    assert second.lm_head.weight.data_ptr() == second.token_embedding.weight.data_ptr()
    assert torch.allclose(first(torch.tensor([[1, 2]])).detach(), second(torch.tensor([[1, 2]])).detach())
