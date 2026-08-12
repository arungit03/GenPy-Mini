import torch


def test_clip_grad_norm_returns_finite_norm_and_obeys_limit():
    parameter = torch.nn.Parameter(torch.ones(4))
    parameter.grad = torch.full_like(parameter, 100.0)
    norm = torch.nn.utils.clip_grad_norm_([parameter], 1.0)
    assert torch.isfinite(norm)
    assert parameter.grad.norm().item() <= 1.00001
