from genpy.training.state import TrainingState


def test_training_state_roundtrip() -> None:
    state = TrainingState(global_step=3, tokens_seen=96, best_validation_loss=1.2)
    assert TrainingState.from_dict(state.to_dict()) == state
