import torch

from genpy.model import CausalSelfAttention


def test_attention_shape_bias_free_and_eager_sdpa_agree() -> None:
    torch.manual_seed(1)
    eager = CausalSelfAttention(16, 4, 4, 8, backend="eager")
    sdpa = CausalSelfAttention(16, 4, 4, 8, backend="sdpa")
    sdpa.load_state_dict(eager.state_dict())
    eager.eval(); sdpa.eval()
    x = torch.randn(2, 5, 16)
    assert eager(x).shape == (2, 5, 16)
    assert torch.allclose(eager(x), sdpa(x), atol=1e-5, rtol=1e-5)
    for module in (eager.q_proj, eager.k_proj, eager.v_proj, eager.o_proj):
        assert module.bias is None
