import pytest
import torch

from genpy.model import RMSNorm, RotaryEmbedding, GenPyForCausalLM
from genpy.verification import causal_lm_loss, gradient_report
from tests.test_model_architecture import tiny_config


@pytest.mark.parametrize("scale", (1.0, 1e-5, 1e3))
def test_rmsnorm_is_finite_at_multiple_input_scales(scale):
    norm = RMSNorm(16)
    x = (scale * torch.randn(2, 4, 16)).requires_grad_()
    output = norm(x)
    output.square().mean().backward()
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
    assert norm.weight.grad is not None
    assert torch.isfinite(norm.weight.grad).all()


def test_rope_and_tiny_model_loss_are_finite():
    rope = RotaryEmbedding(16, 32)
    q = torch.randn(2, 4, 8, 16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    rotated_q, rotated_k = rope(q, k)
    assert torch.isfinite(rotated_q).all() and torch.isfinite(rotated_k).all()
    (rotated_q.square().mean() + rotated_k.square().mean()).backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()

    model = GenPyForCausalLM(tiny_config())
    input_ids = torch.randint(0, 256, (2, 16))
    loss = causal_lm_loss(model(input_ids), input_ids)
    assert torch.isfinite(loss)
    loss.backward()
    assert not gradient_report(model)["non_finite"]


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="CUDA BF16 unavailable")
def test_bfloat16_tiny_forward_is_finite_when_supported():
    model = GenPyForCausalLM(tiny_config()).cuda()
    input_ids = torch.randint(0, 256, (1, 8), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids)
        loss = causal_lm_loss(logits, input_ids)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
