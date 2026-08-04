from __future__ import annotations

from genpy.training.packed_format import validate_packed_manifest
from genpy.training.packing import load_packing_config
from genpy.training.smoke import micro_overfit, smoke_forward


def test_phase4_safe_fixture_pipeline(phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    packing = load_packing_config(
        phase4_fixture["packing_config"], phase4_fixture["root"]
    )
    validation = validate_packed_manifest(
        packing.output_root / "manifests/packing_manifest.json",
        str(packing.tokenizer["fingerprint"]),
        packing.config_hash,
    )
    assert validation["passed"] and len(validation["groups"]) == 6
    forward = smoke_forward(
        phase4_fixture["model_config"], phase4_fixture["packing_config"]
    )
    assert forward["gradients_finite"] and forward["parameter_changed"]
    overfit = micro_overfit(
        phase4_fixture["model_config"],
        phase4_fixture["packing_config"],
        maximum_steps=8,
        timeout_seconds=60,
    )
    assert overfit["final_loss"] < overfit["initial_loss"]
