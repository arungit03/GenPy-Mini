from dataclasses import asdict

import pytest
import torch

from genpy.inference import GenerationConfig, generate, load_checkpoint_weights
from genpy.inference.generation import _apply_repetition_penalty, _filter_logits
from genpy.model import GenPyForCausalLM
from tests.test_model_architecture import tiny_config


class FixedLogitModel(torch.nn.Module):
    def __init__(self, config, token_id):
        super().__init__()
        self.config = config
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.token_id = token_id
        self.seen_lengths = []

    def forward(self, input_ids):
        self.seen_lengths.append(input_ids.shape[1])
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.config.vocab_size),
            -10.0,
            device=input_ids.device,
        )
        logits[..., self.token_id] = 10.0
        return logits


def test_greedy_decoding_and_max_new_tokens_cpu():
    torch.manual_seed(3)
    model = GenPyForCausalLM(tiny_config()).eval()
    prompt = torch.tensor([[4, 5, 6]])
    with torch.inference_mode():
        expected = model(prompt)[:, -1].argmax(dim=-1)
    result = generate(model, prompt, GenerationConfig(max_new_tokens=1, greedy=True))
    assert result.shape == (1, 4)
    assert result[0, -1].item() == expected.item()


def test_seeded_sampling_is_reproducible():
    torch.manual_seed(4)
    model = GenPyForCausalLM(tiny_config()).eval()
    config = GenerationConfig(max_new_tokens=6, temperature=0.8, top_k=20, top_p=0.9, seed=42)
    first = generate(model, torch.tensor([[7, 8]]), config)
    second = generate(model, torch.tensor([[7, 8]]), config)
    assert torch.equal(first, second)


def test_generation_argument_validation():
    with pytest.raises(ValueError):
        GenerationConfig(temperature=0)
    with pytest.raises(ValueError):
        GenerationConfig(top_k=-1)
    with pytest.raises(ValueError):
        GenerationConfig(top_p=0)
    with pytest.raises(ValueError):
        GenerationConfig(repetition_penalty=0)
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=-1)


def test_top_k_and_top_p_filtering():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    top_k = _filter_logits(logits.clone(), GenerationConfig(top_k=1))
    assert torch.isfinite(top_k[0, 0])
    assert torch.isneginf(top_k[0, 1:]).all()
    top_p = _filter_logits(logits.clone(), GenerationConfig(top_p=0.7))
    assert torch.isfinite(top_p[0, 0])
    assert torch.isneginf(top_p[0, 2:]).all()


def test_repetition_penalty_changes_seen_token_logits():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    _apply_repetition_penalty(logits, torch.tensor([[0, 1]]), 2.0)
    assert logits.tolist() == [[1.0, -4.0, 1.0]]


def test_eos_stops_generation():
    config = tiny_config()
    model = FixedLogitModel(config, token_id=2)
    result = generate(
        model,
        torch.tensor([[5]]),
        GenerationConfig(max_new_tokens=10, eos_token_id=2, greedy=True),
    )
    assert result.tolist() == [[5, 2]]


def test_context_is_truncated_to_model_limit():
    config = tiny_config()
    model = FixedLogitModel(config, token_id=5)
    prompt = torch.arange(40).reshape(1, 40) % config.vocab_size
    result = generate(model, prompt, GenerationConfig(max_new_tokens=2, greedy=True))
    assert result.shape[1] == 42
    assert model.seen_lengths == [config.max_seq_len, config.max_seq_len]


def test_strict_checkpoint_weight_loading(tmp_path):
    config = tiny_config()
    source = GenPyForCausalLM(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": source.state_dict(), "model_config": asdict(config)}, checkpoint)
    target = GenPyForCausalLM(config)
    load_checkpoint_weights(target, checkpoint, config)
    assert all(torch.equal(left, right) for left, right in zip(source.parameters(), target.parameters()))
    broken = tmp_path / "broken.pt"
    state = source.state_dict()
    state.pop(next(iter(state)))
    torch.save({"model": state, "model_config": asdict(config)}, broken)
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint_weights(GenPyForCausalLM(config), broken, config)
