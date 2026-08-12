import pytest
import torch

from genpy.config import load_model_config
from genpy.model import GenPyForCausalLM, RotaryEmbedding
from genpy.verification import causal_lm_loss, gradient_report, parameter_finite_report
from genpy.verification.diagnostics import model_diagnostics
from tests.test_model_architecture import tiny_config


def test_integrated_tiny_model_verification_contract():
    torch.manual_seed(14)
    model = GenPyForCausalLM(tiny_config()).eval()
    input_ids = torch.randint(0, 256, (2, 16))
    logits = model(input_ids)
    loss = causal_lm_loss(logits, input_ids)
    assert logits.shape == (2, 16, 256)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert gradient_report(model)["missing"] == []
    assert gradient_report(model)["non_finite"] == []
    assert all(report.is_finite for report in parameter_finite_report(model))
    diagnostics = model_diagnostics(model)
    assert diagnostics["weight_tied"] is True
    assert diagnostics["parameters_finite"] is True


def test_one_sgd_like_update_keeps_tied_parameter_storage():
    model = GenPyForCausalLM(tiny_config())
    input_ids = torch.randint(0, 256, (1, 8))
    loss = causal_lm_loss(model(input_ids), input_ids)
    loss.backward()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-1e-3)
    assert model.lm_head.weight is model.token_embedding.weight
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_production_rope_context_limit():
    config = load_model_config("configs/model_200m.yaml")
    rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
    cos, sin = rope.get_cos_sin(1024)
    assert cos.shape[0] == 1024 and sin.shape[0] == 1024
    with pytest.raises(ValueError, match="exceeds"):
        rope.get_cos_sin(1025)
