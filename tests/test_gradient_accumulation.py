import torch

from genpy.config import TrainingConfig
from genpy.model import GenPyForCausalLM
from genpy.training import TrainingEngine, create_dataloader, PackedTokenDataset
from tests.test_training_data import write_tokens
from tests.test_model_architecture import tiny_config


def test_accumulation_reaches_one_optimizer_step(tmp_path):
    config = TrainingConfig(1, 1, 2, 1e-3, 1e-4, 0.0, 0.0, 1.0, "fp32", 100, 100, 100, "checkpoints", "logs", 4, 0.9, 0.95, 1e-8, 0, False, 1, 2)
    dataset = PackedTokenDataset(write_tokens(tmp_path, range(32)), 4)
    loader, sampler = create_dataloader(dataset, 1, shuffle=False)
    model = GenPyForCausalLM(tiny_config())
    engine = TrainingEngine(model, config, loader, max_steps=1, train_sampler=sampler)
    result = engine.train()
    assert result["global_step"] == 1
    assert engine.state.micro_step == 2
    assert engine.optimizer.state
    assert all(parameter.grad is None for parameter in model.parameters())
