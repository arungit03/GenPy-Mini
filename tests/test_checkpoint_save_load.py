from training_helpers import tiny_engine


def test_checkpoint_save_load_and_latest_pointer(tmp_path) -> None:
    engine = tiny_engine(tmp_path); engine.train_optimizer_step(); saved = engine.save_checkpoint()
    assert (saved / "COMPLETE").is_file() and engine.checkpoints.latest() == saved
    restored = tiny_engine(tmp_path / "restored")
    state = restored.checkpoints.load(saved, restored.model, restored.optimizer, restored.scheduler, restored.precision, restored.state, restored.train_batcher, restored.metadata)
    assert state.global_step == 1 and restored.model.lm_head.weight.data_ptr() == restored.model.token_embedding.weight.data_ptr()
    engine.train_dataset.close(); engine.validation_dataset.close(); restored.train_dataset.close(); restored.validation_dataset.close()
