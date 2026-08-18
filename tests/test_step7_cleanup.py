import json

import torch

from genpy.training import TrainingEngine
from scripts.training_smoke_test import build, configs
from tests.test_checkpoint import setup_engine


def test_resumed_logging_uses_cumulative_training_state_tokens(tmp_path):
    model_config, train_config = configs()
    engine, _ = build(tmp_path / "run", model_config, train_config, 3)
    engine.train(stop_after_steps=1)
    checkpoint = engine.save_checkpoint()

    resumed, _ = build(tmp_path / "run", model_config, train_config, 3)
    resumed.load_checkpoint(checkpoint)
    resumed.train()

    records = [json.loads(line) for line in (tmp_path / "run" / "logs" / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[-1]["global_step"] == 3
    assert records[-1]["tokens_seen"] == resumed.state.tokens_seen == 96


def test_repeated_validation_is_deterministic_and_does_not_advance_sampler(tmp_path):
    engine, _ = setup_engine(tmp_path)
    validation_dir = tmp_path / "validation"
    from tests.test_training_data import write_tokens
    from genpy.training import create_dataloader, PackedTokenDataset

    validation_dataset = PackedTokenDataset(write_tokens(validation_dir, range(32)), 4)
    validation_loader, validation_sampler = create_dataloader(validation_dataset, 1, shuffle=False)
    engine.validation_loader = validation_loader
    before = validation_sampler.state_dict()
    first = engine.validate(max_batches=2)
    middle = validation_sampler.state_dict()
    second = engine.validate(max_batches=2)
    after = validation_sampler.state_dict()
    assert first == second
    assert before == middle == after
    validation_dataset.close()


def test_resume_trajectory_regression_remains_exact(tmp_path):
    model_config, train_config = configs()
    torch.manual_seed(99)
    continuous, _ = build(tmp_path / "continuous", model_config, train_config, 3)
    continuous.train()
    torch.manual_seed(99)
    split, _ = build(tmp_path / "split", model_config, train_config, 3)
    split.train(stop_after_steps=2)
    checkpoint = split.save_checkpoint()
    resumed, _ = build(tmp_path / "split", model_config, train_config, 3)
    resumed.load_checkpoint(checkpoint)
    resumed.train()
    assert resumed.state.tokens_seen == continuous.state.tokens_seen
    assert resumed.scheduler.get_last_lr() == continuous.scheduler.get_last_lr()
    assert all(torch.equal(left, right) for left, right in zip(continuous.model.parameters(), resumed.model.parameters()))
