import torch

from genpy.config import ModelConfig, TrainingConfig
from genpy.model import GenPyForCausalLM
from genpy.training import PackedTokenDataset, TrainingEngine, create_dataloader
from tests.test_training_data import write_tokens


def test_production_engine_construction_without_training(tmp_path):
    model_config = ModelConfig("tiny-engine", 64, 16, 32, 2, 4, 8, 64, 1e-5, 10000.0, True)
    train_config = TrainingConfig(1, 1, 1, 1e-3, 1e-4, 0.0, 0.0, 1.0, "fp32", 1, 100, 100, "checkpoints", "logs", 4, 0.9, 0.95, 1e-8, 0, False, 1, 2)
    dataset = PackedTokenDataset(write_tokens(tmp_path, range(32)), 4)
    loader, sampler = create_dataloader(dataset, 1, shuffle=False)
    model = GenPyForCausalLM(model_config)
    engine = TrainingEngine(model, train_config, loader, max_steps=2, train_sampler=sampler)
    assert engine.state.global_step == 0
    assert sum(parameter.numel() for parameter in engine.model.parameters()) > 0
    assert torch.isfinite(torch.tensor(engine.scheduler.get_last_lr()[0]))
