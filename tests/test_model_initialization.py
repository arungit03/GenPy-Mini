from dataclasses import replace

import torch

from genpy.config import load_config
from genpy.model import GenPyForCausalLM, RMSNorm
from genpy.utils.reproducibility import set_seed


def test_random_initialization_is_reproducible_and_finite() -> None:
    config = replace(load_config("configs/model_200m.yaml").model, vocab_size=64, max_seq_len=16, n_layers=1, d_model=16, n_heads=2, head_dim=8, ffn_hidden_size=24)
    set_seed(42)
    first = GenPyForCausalLM(config)
    set_seed(42)
    second = GenPyForCausalLM(config)
    assert torch.equal(first.token_embedding.weight, second.token_embedding.weight)
    assert all(torch.isfinite(parameter).all() for parameter in first.parameters())
    assert all(torch.allclose(module.weight, torch.ones_like(module.weight)) for module in first.modules() if isinstance(module, RMSNorm))
    assert all(module.bias is None for module in first.modules() if isinstance(module, torch.nn.Linear))


def test_tiny_backward_has_finite_gradients() -> None:
    config = replace(load_config("configs/model_200m.yaml").model, vocab_size=64, max_seq_len=16, n_layers=1, d_model=16, n_heads=2, head_dim=8, ffn_hidden_size=24)
    model = GenPyForCausalLM(config)
    loss = model(torch.randint(0, 64, (1, 4))).float().mean()
    loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()
