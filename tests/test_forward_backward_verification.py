from _verification_cli import tiny_config
from genpy.model import GenPyForCausalLM
from genpy.verification.backward import run_backward_smoke
from genpy.verification.forward import run_forward_shapes


def test_forward_backward_helpers() -> None:
    model = GenPyForCausalLM(tiny_config(), attention_backend="eager")
    assert run_forward_shapes(model, 128)["passed"]
    assert run_backward_smoke(model, 128)["passed"]
