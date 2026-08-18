import math

import pytest
import torch
from torch.nn import functional as F

from genpy.evaluation import evaluate_packed_dataset
from genpy.evaluation.evaluation import _devices_match
from genpy.model import GenPyForCausalLM
from genpy.training.data import PackedTokenDataset
from tests.test_model_architecture import tiny_config
from tests.test_training_data import write_tokens


def test_cuda_aliases_resolve_to_the_same_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert _devices_match(torch.device("cuda:0"), "cuda")
    assert _devices_match(torch.device("cuda:0"), "cuda:0")


def test_genuinely_mismatched_cuda_devices_are_rejected():
    assert not _devices_match(torch.device("cuda:0"), "cuda:1")


def test_cpu_device_matching_and_mismatch_are_unchanged():
    assert _devices_match(torch.device("cpu"), "cpu")
    assert not _devices_match(torch.device("cpu"), "cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_alias_is_accepted_by_evaluation_when_available(tmp_path):
    model = GenPyForCausalLM(tiny_config()).to("cuda:0").eval()
    dataset = PackedTokenDataset(write_tokens(tmp_path, range(9)), sequence_length=4)
    result = evaluate_packed_dataset(model, dataset, device="cuda")
    assert result.evaluation_windows == 2
    assert result.evaluated_tokens == 8


def test_evaluation_is_token_weighted_and_reports_windows(tmp_path):
    torch.manual_seed(8)
    model = GenPyForCausalLM(tiny_config()).eval()
    dataset = PackedTokenDataset(write_tokens(tmp_path, range(21)), sequence_length=4)
    result = evaluate_packed_dataset(model, dataset, device="cpu", batch_size=2)
    with torch.inference_mode():
        total = 0.0
        for index in range(len(dataset)):
            batch = dataset[index]
            logits = model(batch["input_ids"][None]).float()
            total += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch["targets"],
                reduction="sum",
            ).item()
    assert result.evaluation_windows == 5
    assert result.evaluated_tokens == 20
    assert result.sequence_length == 4
    assert result.validation_loss == pytest.approx(total / 20)
    assert result.perplexity == pytest.approx(math.exp(result.validation_loss))


def test_empty_or_too_small_evaluation_dataset_is_rejected(tmp_path):
    model = GenPyForCausalLM(tiny_config()).eval()
    dataset = PackedTokenDataset(write_tokens(tmp_path, range(4)), sequence_length=4)
    assert len(dataset) == 0
    with pytest.raises(ValueError, match="no complete windows"):
        evaluate_packed_dataset(model, dataset)
