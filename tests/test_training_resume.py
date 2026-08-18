import torch

from scripts.training_smoke_test import build, configs


def test_deterministic_resume_matches_continuous_training(tmp_path):
    model_config, train_config = configs()
    torch.manual_seed(99)
    continuous, _ = build(tmp_path / "continuous", model_config, train_config, 3)
    continuous.train()
    torch.manual_seed(99)
    split, manager = build(tmp_path / "split", model_config, train_config, 3)
    split.train(stop_after_steps=2)
    checkpoint = split.save_checkpoint()
    resumed, _ = build(tmp_path / "split", model_config, train_config, 3)
    resumed.load_checkpoint(checkpoint)
    resumed.train()
    assert resumed.state.global_step == continuous.state.global_step
    assert resumed.state.tokens_seen == continuous.state.tokens_seen
    assert resumed.scheduler.get_last_lr() == continuous.scheduler.get_last_lr()
    assert all(torch.equal(a, b) for a, b in zip(continuous.model.parameters(), resumed.model.parameters()))
