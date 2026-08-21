from dataclasses import replace

import torch

from genpy.config import load_config
from genpy.model import RMSNorm, SwiGLU, TransformerBlock


def tiny_config():
    return replace(load_config("configs/model_200m.yaml").model, vocab_size=128, max_seq_len=32, n_layers=2, d_model=32, n_heads=4, head_dim=8, ffn_hidden_size=48)


def test_pre_norm_block_shape_and_modules() -> None:
    block = TransformerBlock(tiny_config(), attention_backend="eager")
    x = torch.randn(2, 5, 32)
    assert block(x).shape == x.shape
    assert isinstance(block.attn_norm, RMSNorm)
    assert isinstance(block.ffn_norm, RMSNorm)
    assert isinstance(block.mlp, SwiGLU)
