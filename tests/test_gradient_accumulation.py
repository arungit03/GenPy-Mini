from training_helpers import tiny_engine


def test_optimizer_step_occurs_after_accumulation(tmp_path) -> None:
    engine = tiny_engine(tmp_path, accumulation=2)
    before = engine.state.global_step
    result = engine.train_optimizer_step()
    assert before == 0 and engine.state.global_step == 1 and result["tokens_seen"] == 32
    engine.train_dataset.close(); engine.validation_dataset.close()
