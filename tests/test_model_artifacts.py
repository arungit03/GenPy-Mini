from pathlib import Path

from genpy.training.artifacts import save_model_artifact, verify_saved_model_artifact, verify_model
from training_helpers import tiny_engine


def test_model_only_artifact_save_reload_and_tying(tmp_path) -> None:
    engine = tiny_engine(tmp_path, steps=1)
    expected = verify_model(engine.model, expected_parameters=sum(parameter.numel() for parameter in engine.model.parameters()))["parameter_count"]
    output = save_model_artifact(engine.model, tmp_path / "artifact", "configs/model_200m.yaml", "data/tokenized/genpy-32k/TOKEN_CACHE_MANIFEST.json", {"status": "test"}, expected_parameters=expected)
    assert output.is_file()
    result = verify_saved_model_artifact(engine.model, output.parent, expected_parameters=expected)
    assert result["save_reload_pass"] and result["weight_tying"]
