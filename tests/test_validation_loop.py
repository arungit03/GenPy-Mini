from training_helpers import tiny_engine


def test_validation_is_finite_and_restores_train_mode(tmp_path) -> None:
    engine = tiny_engine(tmp_path); engine.model.train()
    value = engine.validate()
    assert value == value and engine.model.training
    engine.train_dataset.close(); engine.validation_dataset.close()
