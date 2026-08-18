import math

import pytest
import torch
from torch.nn import functional as F

from genpy.evaluation import evaluate_packed_dataset
from genpy.model import GenPyForCausalLM
from genpy.training.data import PackedTokenDataset
from tests.test_model_architecture import tiny_config
from tests.test_training_data import write_tokens


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
