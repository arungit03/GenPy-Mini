from dataclasses import replace

import pytest
import torch

from genpy.config import load_config
from genpy.model import GenPyForCausalLM


def tiny_model() -> GenPyForCausalLM:
    config = replace(load_config("configs/model_200m.yaml").model, vocab_size=128, max_seq_len=32, n_layers=2, d_model=32, n_heads=4, head_dim=8, ffn_hidden_size=48)
    return GenPyForCausalLM(config, attention_backend="eager")


def test_tiny_forward_shape_finiteness_and_guards() -> None:
    model = tiny_model().eval()
    logits = model(torch.randint(0, 128, (2, 16)))
    assert logits.shape == (2, 16, 128)
    assert torch.isfinite(logits).all()
    with pytest.raises(ValueError, match="context"):
        model(torch.zeros(1, 33, dtype=torch.long))
    with pytest.raises(ValueError, match="vocabulary"):
        model(torch.full((1, 2), 128, dtype=torch.long))
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 3, 4, dtype=torch.long))
