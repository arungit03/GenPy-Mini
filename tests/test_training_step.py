import torch

from genpy.config import TrainingConfig
from genpy.model import GenPyForCausalLM
from genpy.training import PackedTokenDataset, TrainingEngine, create_dataloader
from tests.test_model_architecture import tiny_config
from tests.test_training_data import write_tokens


def test_training_step_updates_model_and_scheduler(tmp_path):
    config = TrainingConfig(1, 1, 1, 1e-3, 1e-4, 0.0, 0.0, 1.0, "fp32", 1, 100, 100, "checkpoints", "logs", 4, 0.9, 0.95, 1e-8, 0, False, 1, 2)
    dataset = PackedTokenDataset(write_tokens(tmp_path, range(32)), 4)
    loader, _ = create_dataloader(dataset, 1, shuffle=False)
    model = GenPyForCausalLM(tiny_config())
    before = model.token_embedding.weight.detach().clone()
    engine = TrainingEngine(model, config, loader, max_steps=1)
    result = engine.train()
    assert result["final_loss"] is not None and torch.isfinite(torch.tensor(result["final_loss"]))
    assert not torch.equal(before, model.token_embedding.weight)
    assert engine.scheduler.step_count == 1
