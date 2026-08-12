import torch

from genpy.config import TrainingConfig
from genpy.model import GenPyForCausalLM
from genpy.training import PackedTokenDataset, TrainingEngine, create_dataloader
from tests.test_model_architecture import tiny_config
from tests.test_training_data import write_tokens


def test_validation_has_no_gradients_and_restores_training_mode(tmp_path):
    config = TrainingConfig(1, 1, 1, 1e-3, 1e-4, 0.0, 0.0, 1.0, "fp32", 1, 100, 100, "checkpoints", "logs", 4, 0.9, 0.95, 1e-8, 0, False, 2, 2)
    train = PackedTokenDataset(write_tokens(tmp_path, range(32)), 4)
    valid = PackedTokenDataset(write_tokens(tmp_path / "valid", range(32)), 4)
    train_loader, _ = create_dataloader(train, 1, shuffle=False)
    valid_loader, _ = create_dataloader(valid, 1, shuffle=False)
    model = GenPyForCausalLM(tiny_config()).train()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    engine = TrainingEngine(model, config, train_loader, valid_loader, max_steps=1)
    loss = engine.validate(max_batches=2)
    assert torch.isfinite(torch.tensor(loss))
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
