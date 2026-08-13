import json
from contextlib import nullcontext

import pytest
import torch

from genpy.config import TrainingConfig
from genpy.model import GenPyForCausalLM
from genpy.training import PackedTokenDataset, TrainingEngine, create_dataloader
from genpy.training.logger import TrainingLogger
from genpy.training.precision import PrecisionManager
from tests.test_model_architecture import tiny_config
from tests.test_training_data import write_tokens


def training_config(precision="fp32"):
    return TrainingConfig(
        1, 1, 1, 1e-3, 1e-4, 0.0, 0.0, 1.0, precision,
        1, 100, 100, "checkpoints", "logs", 4, 0.9, 0.95, 1e-8,
        0, False, 1, 2,
    )


class CPUGradScalerPrecision(PrecisionManager):
    """Exercise the real GradScaler on CPU for deterministic unit tests."""

    def __init__(self, *, inject_overflow=False, mode="fp16"):
        self.device = torch.device("cpu")
        self.mode = mode
        self.inject_overflow = inject_overflow
        self.injected = False
        self.scaler = torch.amp.GradScaler("cpu", enabled=mode == "fp16")

    def autocast(self):
        return nullcontext()

    def unscale_(self, optimizer):
        super().unscale_(optimizer)

    def backward(self, loss):
        super().backward(loss)
        if self.inject_overflow and not self.injected:
            for parameter in self._optimizer.param_groups[0]["params"]:
                if parameter.grad is not None:
                    parameter.grad.fill_(float("inf"))
                    self.injected = True
                    break

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer


def make_engine(tmp_path, precision, *, max_steps=1):
    dataset = PackedTokenDataset(write_tokens(tmp_path / "data", range(64)), 4)
    loader, sampler = create_dataloader(dataset, 1, shuffle=False)
    model = GenPyForCausalLM(tiny_config())
    logger = TrainingLogger(tmp_path / "logs")
    engine = TrainingEngine(
        model,
        training_config("fp32"),
        loader,
        max_steps=max_steps,
        precision=precision,
        logger=logger,
        train_sampler=sampler,
    )
    precision.set_optimizer(engine.optimizer)
    return engine, dataset


def test_fp16_overflow_skips_update_and_reduces_scaler(tmp_path):
    precision = CPUGradScalerPrecision(inject_overflow=True)
    engine, dataset = make_engine(tmp_path, precision)
    before = engine.model.token_embedding.weight.detach().clone()
    initial_scale = precision.scaler.get_scale()
    result = engine.train()
    assert result["global_step"] == 1
    assert engine.state.optimizer_steps == 1
    assert engine.state.micro_step == 2
    assert engine.state.tokens_seen == 8
    assert engine.scheduler.step_count == 1
    assert precision.scaler.get_scale() < initial_scale
    assert not torch.equal(before, engine.model.token_embedding.weight)
    assert all(parameter.grad is None for parameter in engine.model.parameters())
    records = [json.loads(line) for line in (tmp_path / "logs" / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    overflow = next(record for record in records if record.get("fp16_overflow"))
    assert overflow["skipped_update"] is True
    assert overflow["scaler_scale"] == precision.scaler.get_scale()
    dataset.close()


def test_normal_fp16_update_uses_scaler_and_advances_normally(tmp_path):
    precision = CPUGradScalerPrecision()
    engine, dataset = make_engine(tmp_path, precision)
    initial_scale = precision.scaler.get_scale()
    result = engine.train()
    assert result["global_step"] == 1
    assert engine.state.optimizer_steps == 1
    assert engine.scheduler.step_count == 1
    assert precision.scaler.get_scale() == initial_scale
    dataset.close()


@pytest.mark.parametrize("mode", ["fp32", "bf16"])
def test_nonfinite_gradients_still_fail_without_grad_scaler(tmp_path, mode):
    precision = CPUGradScalerPrecision(mode=mode, inject_overflow=True)
    engine, dataset = make_engine(tmp_path, precision)
    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        engine.train()
    assert engine.state.global_step == 0
    assert engine.state.optimizer_steps == 0
    assert engine.state.micro_step == 1
    assert engine.state.tokens_seen == 4
    dataset.close()


def test_fp16_scaler_state_round_trips_exactly():
    first = CPUGradScalerPrecision()
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    first.scaler.scale(parameter * 2).backward()
    first.unscale_(optimizer)
    first.step(optimizer)
    state = first.state_dict()
    second = CPUGradScalerPrecision()
    second.load_state_dict(state)
    assert second.state_dict() == state
