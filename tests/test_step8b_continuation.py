import pytest
import torch

from genpy.config import ModelConfig, TrainingConfig
from genpy.model import GenPyForCausalLM
from genpy.training import PackedTokenDataset, TrainingEngine, create_dataloader
from genpy.training.checkpoint import CheckpointManager
from tests.test_model_architecture import tiny_config
from tests.test_training_data import write_tokens


def config(lr=1e-3):
    return TrainingConfig(3, 1, 1, lr, lr / 10, 0.1, 0.1, 1.0, "fp32", 10, 100, 100, "checkpoints", "logs", 4, 0.9, 0.95, 1e-8, 0, False, 2, 2)


def engine(tmp_path, model_config, train_config, token_values):
    dataset = PackedTokenDataset(write_tokens(tmp_path / "data", token_values), 4)
    loader, sampler = create_dataloader(dataset, 1, seed=7, shuffle=True)
    manager = CheckpointManager(tmp_path / "checkpoints", keep_last=2)
    return TrainingEngine(GenPyForCausalLM(model_config), train_config, loader, max_steps=5, checkpoint_manager=manager, train_sampler=sampler), dataset


def make_source_checkpoint(tmp_path):
    source, dataset = engine(tmp_path / "source", tiny_config(), config(), range(32))
    source.state.global_step = 6297
    source.state.optimizer_steps = 6297
    source.state.tokens_seen = 123456
    source_path = source.save_checkpoint()
    dataset.close()
    return source_path


def test_init_from_checkpoint_loads_weights_only_and_starts_fresh(tmp_path):
    source_path = make_source_checkpoint(tmp_path)
    target, dataset = engine(tmp_path / "target", tiny_config(), config(5e-5), range(48))
    initial_sampler = target.train_sampler.state_dict()
    provenance = target.initialize_from_checkpoint(source_path)
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    for actual, expected in zip(target.model.parameters(), payload["model"].values()):
        assert torch.equal(actual, expected)
    assert target.state.global_step == 0
    assert target.state.optimizer_steps == 0
    assert target.state.tokens_seen == 0
    assert target.state.micro_step == 0
    assert target.train_sampler.state_dict() == initial_sampler
    assert target.scheduler.step_count == 0
    assert target.scheduler.get_last_lr()[0] == pytest.approx(5e-5)
    assert provenance["initialization_mode"] == "weights_only"
    assert provenance["source_global_step"] == 6297
    assert provenance["source_tokens_seen"] == 123456
    assert provenance["optimizer_restored"] is False
    assert provenance["scaler_restored"] is False
    target_path = target.save_checkpoint()
    continuation_payload = torch.load(target_path, map_location="cpu", weights_only=False)
    assert continuation_payload["provenance"] == provenance
    dataset.close()


def test_exact_resume_rejects_different_dataset_sampler_state(tmp_path):
    source_path = make_source_checkpoint(tmp_path)
    target, dataset = engine(tmp_path / "different", tiny_config(), config(), range(48))
    with pytest.raises(ValueError, match="sampler mismatch for dataset_size"):
        target.load_checkpoint(source_path)
    dataset.close()


def test_init_from_checkpoint_rejects_incompatible_architecture(tmp_path):
    source_path = make_source_checkpoint(tmp_path)
    incompatible = ModelConfig("incompatible", 64, 16, 32, 2, 4, 8, 32, 1e-5, 10000.0, True)
    target, dataset = engine(tmp_path / "incompatible", incompatible, config(5e-5), range(32))
    with pytest.raises(ValueError, match="model configuration is incompatible"):
        target.initialize_from_checkpoint(source_path)
    dataset.close()
