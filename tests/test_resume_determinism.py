from training_helpers import tiny_engine


def test_engine_resume_continues_step_and_tokens(tmp_path) -> None:
    engine = tiny_engine(tmp_path); engine.train_optimizer_step(); engine.save_checkpoint()
    resumed = tiny_engine(tmp_path / "new"); state = resumed.resume(tmp_path / "run" / "checkpoints" / "step_000000000001")
    assert state.global_step == 1 and resumed.state.tokens_seen == 16
    engine.train_dataset.close(); engine.validation_dataset.close(); resumed.train_dataset.close(); resumed.validation_dataset.close()
