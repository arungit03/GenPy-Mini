from dataclasses import replace

import torch

from genpy.config import load_config
from genpy.model import GenPyForCausalLM


def test_future_tokens_do_not_change_earlier_outputs() -> None:
    base = replace(load_config("configs/model_200m.yaml").model, vocab_size=128, max_seq_len=16, n_layers=2, d_model=32, n_heads=4, head_dim=8, ffn_hidden_size=48)
    model = GenPyForCausalLM(base, attention_backend="eager").eval()
    first = torch.tensor([[10, 20, 30, 40]])
    second = torch.tensor([[10, 20, 30, 99]])
    with torch.no_grad():
        left, right = model(first), model(second)
    assert torch.allclose(left[:, :3], right[:, :3], atol=1e-5, rtol=1e-5)
