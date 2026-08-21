from _verification_cli import tiny_config
from genpy.model import GenPyForCausalLM
from genpy.verification.backward import run_backward_smoke


def test_gradient_coverage_includes_core_components() -> None:
    result = run_backward_smoke(GenPyForCausalLM(tiny_config(), attention_backend="eager"), 128)
    assert result["gradient_audit"]["non_finite_gradients"] == []
    assert result["gradient_audit"]["parameters_without_gradients"] == 0
    assert all(result["representative_gradients"].values())
