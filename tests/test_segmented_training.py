import pytest
import torch

from tests.test_model_architecture import tiny_config
from scripts.training_smoke_test import build, configs


def test_fresh_segment_stops_after_additional_steps_and_keeps_full_horizon(tmp_path):
    model_config, train_config = configs()
    engine, _ = build(tmp_path / "fresh", model_config, train_config, 20)
    result = engine.train(stop_after_steps=5)
    assert result["global_step"] == 5
    assert engine.state.global_step == 5
    assert engine.max_steps == 20
    assert engine.scheduler.total_steps == 20
    assert engine.scheduler.step_count == 5
    engine.train_loader.dataset.close()


def test_resume_segment_advances_cumulative_steps_and_preserves_horizon(tmp_path):
    model_config, train_config = configs()
    first, _ = build(tmp_path / "run", model_config, train_config, 20)
    first.train(stop_after_steps=5)
    checkpoint = first.save_checkpoint()

    resumed, _ = build(tmp_path / "run", model_config, train_config, 20)
    resumed.load_checkpoint(checkpoint)
    result = resumed.train(stop_after_steps=5)
    assert result["global_step"] == 10
    assert resumed.state.global_step == 10
    assert resumed.max_steps == 20
    assert resumed.scheduler.total_steps == 20
    assert resumed.scheduler.step_count == 10
    resumed.train_loader.dataset.close()


def test_final_segment_clamps_to_full_max_steps(tmp_path):
    model_config, train_config = configs()
    first, _ = build(tmp_path / "run", model_config, train_config, 20)
    first.train(stop_after_steps=18)
    checkpoint = first.save_checkpoint()

    resumed, _ = build(tmp_path / "run", model_config, train_config, 20)
    resumed.load_checkpoint(checkpoint)
    result = resumed.train(stop_after_steps=5)
    assert result["global_step"] == 20
    assert resumed.state.global_step == 20
    assert resumed.scheduler.total_steps == 20
    assert resumed.scheduler.step_count == 20
    resumed.train_loader.dataset.close()


def test_segmented_trajectory_matches_uninterrupted_training(tmp_path):
    model_config, train_config = configs()
    torch.manual_seed(99)
    continuous, _ = build(tmp_path / "continuous", model_config, train_config, 20)
    continuous.train()

    torch.manual_seed(99)
    segmented, _ = build(tmp_path / "segmented", model_config, train_config, 20)
    segmented.train(stop_after_steps=5)
    checkpoint = segmented.save_checkpoint()
    for expected_step in (10, 15):
        resumed, _ = build(tmp_path / "segmented", model_config, train_config, 20)
        resumed.load_checkpoint(checkpoint)
        resumed.train(stop_after_steps=5)
        checkpoint = resumed.save_checkpoint()
    final, _ = build(tmp_path / "segmented", model_config, train_config, 20)
    final.load_checkpoint(checkpoint)
    final.train(stop_after_steps=5)

    assert final.state.global_step == continuous.state.global_step == 20
    assert final.state.tokens_seen == continuous.state.tokens_seen
    assert final.scheduler.state_dict() == continuous.scheduler.state_dict()
    assert all(torch.equal(left, right) for left, right in zip(continuous.model.parameters(), final.model.parameters()))
    continuous.train_loader.dataset.close()
    final.train_loader.dataset.close()


def test_stop_after_steps_must_be_positive(tmp_path):
    model_config, train_config = configs()
    engine, _ = build(tmp_path / "run", model_config, train_config, 20)
    with pytest.raises(ValueError, match="stop_after_steps"):
        engine.train(stop_after_steps=0)
    engine.train_loader.dataset.close()


def test_resume_rejects_mismatched_full_max_steps(tmp_path):
    model_config, train_config = configs()
    source, _ = build(tmp_path / "source", model_config, train_config, 20)
    source.train(stop_after_steps=5)
    checkpoint = source.save_checkpoint()
    target, _ = build(tmp_path / "target", model_config, train_config, 5)
    with pytest.raises(ValueError, match="max_steps"):
        target.load_checkpoint(checkpoint)
    target.train_loader.dataset.close()
