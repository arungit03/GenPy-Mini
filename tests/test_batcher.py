from training_helpers import tiny_engine


def test_random_batcher_state_restores(tmp_path) -> None:
    engine = tiny_engine(tmp_path)
    first = engine.train_batcher.next_batch()
    state = engine.train_batcher.state_dict()
    second = engine.train_batcher.next_batch()
    engine.train_batcher.load_state_dict(state)
    repeated = engine.train_batcher.next_batch()
    assert second[0].equal(repeated[0]) and first[0].shape == (1, 16)
    engine.train_dataset.close(); engine.validation_dataset.close()
