import torch

from genpy.config import TrainingConfig
from genpy.model import GenPyForCausalLM
from genpy.training import PackedTokenDataset, TrainingEngine, create_dataloader
from genpy.training.checkpoint import CheckpointManager
from tests.test_model_architecture import tiny_config
from tests.test_training_data import write_tokens


def setup_engine(tmp_path):
    config = TrainingConfig(1, 1, 1, 1e-3, 1e-4, 0.0, 0.0, 1.0, "fp32", 1, 100, 100, "checkpoints", "logs", 4, 0.9, 0.95, 1e-8, 0, False, 1, 2)
    dataset = PackedTokenDataset(write_tokens(tmp_path / "data", range(32)), 4)
    loader, sampler = create_dataloader(dataset, 1, shuffle=True, seed=2)
    model = GenPyForCausalLM(tiny_config())
    manager = CheckpointManager(tmp_path / "checkpoints", keep_last=2)
    return TrainingEngine(model, config, loader, max_steps=2, checkpoint_manager=manager, train_sampler=sampler), manager


def test_atomic_checkpoint_round_trip_and_latest(tmp_path):
    engine, manager = setup_engine(tmp_path)
    engine.train()
    checkpoint = engine.save_checkpoint()
    assert checkpoint.is_file()
    assert manager.latest_path() == checkpoint
    fresh, _ = setup_engine(tmp_path / "fresh")
    fresh.load_checkpoint(checkpoint)
    assert fresh.state.global_step == engine.state.global_step
    assert fresh.model.lm_head.weight is fresh.model.token_embedding.weight
