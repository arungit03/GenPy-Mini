from pathlib import Path

import torch

from genpy.config import ModelConfig, load_model_config
from genpy.model import GenPyForCausalLM


ROOT = Path(__file__).resolve().parents[1]


def tiny_config() -> ModelConfig:
    return ModelConfig(
        name="tiny",
        vocab_size=256,
        max_seq_len=32,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        intermediate_size=128,
        norm_eps=1e-5,
        rope_theta=10000.0,
        tie_embeddings=True,
    )


def test_tiny_full_model_forward():
    model = GenPyForCausalLM(tiny_config()).eval()
    input_ids = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (2, 16, 256)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_model_initialization_uses_norm_one_and_scaled_residual_outputs():
    config = tiny_config()
    model = GenPyForCausalLM(config)
    expected_std = config.initializer_range / ((2 * config.num_layers) ** 0.5)
    assert torch.allclose(model.final_norm.weight, torch.ones_like(model.final_norm.weight))
    assert abs(model.blocks[0].attention.o_proj.weight.std().item() - expected_std) < 0.004
    assert abs(model.blocks[0].mlp.down_proj.weight.std().item() - expected_std) < 0.004
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_model_rejects_invalid_input_shapes_and_tokens():
    model = GenPyForCausalLM(tiny_config())
    with torch.no_grad():
        try:
            model(torch.zeros(2, 3, 1, dtype=torch.long))
        except ValueError as error:
            assert "rank 2" in str(error)
        else:
            raise AssertionError("rank-3 input was accepted")
        try:
            model(torch.full((1, 2), 256, dtype=torch.long))
        except ValueError as error:
            assert "input_ids" in str(error)
        else:
            raise AssertionError("out-of-range token was accepted")


def test_production_architecture_inspection_and_count():
    config = load_model_config(ROOT / "configs" / "model_200m.yaml")
    model = GenPyForCausalLM(config)
    assert len(model.blocks) == 24
    assert model.config.hidden_size == 768
    assert all(block.attention.num_heads == 12 for block in model.blocks)
    assert all(block.attention.head_dim == 64 for block in model.blocks)
    assert all(block.mlp.gate_proj.weight.shape == (2176, 768) for block in model.blocks)
    assert all(block.mlp.down_proj.weight.shape == (768, 2176) for block in model.blocks)
    assert sum(parameter.numel() for parameter in model.parameters()) == 201_560_832
    linear_modules = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    assert all(module.bias is None for module in linear_modules)
    assert not hasattr(model, "position_embedding")
