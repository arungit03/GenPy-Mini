from dataclasses import replace

import torch

from genpy.config import load_config
from genpy.model import GenPyForCausalLM


def test_tying_and_logits_survive_reload() -> None:
    config = replace(load_config("configs/model_200m.yaml").model, vocab_size=64, max_seq_len=16, n_layers=1, d_model=16, n_heads=2, head_dim=8, ffn_hidden_size=24)
    first = GenPyForCausalLM(config)
    second = GenPyForCausalLM(config)
    state = first.state_dict()
    second.load_state_dict(state)
    ids = torch.tensor([[1, 2, 3]])
    assert second.lm_head.weight.data_ptr() == second.token_embedding.weight.data_ptr()
    assert torch.allclose(first(ids), second(ids))
