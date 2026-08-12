import torch

from genpy.model import GenPyBlock, RMSNorm, SwiGLU


def test_transformer_block_is_pre_norm_with_residuals_and_gradients():
    block = GenPyBlock(
        hidden_size=32,
        num_heads=4,
        head_dim=8,
        intermediate_size=64,
        max_seq_len=16,
    )
    x = torch.randn(2, 6, 32, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    assert isinstance(block.attn_norm, RMSNorm)
    assert isinstance(block.ffn_norm, RMSNorm)
    assert block.attn_norm is not block.ffn_norm
    assert isinstance(block.mlp, SwiGLU)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert all(parameter.grad is not None for parameter in block.parameters() if parameter.requires_grad)
