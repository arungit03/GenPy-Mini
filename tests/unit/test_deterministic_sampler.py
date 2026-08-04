from __future__ import annotations

from genpy.training.sampler import DeterministicSampler, SamplerState


def test_sampler_is_deterministic_rank_separated_and_resumable() -> None:
    data = list(range(20))
    first = list(DeterministicSampler(data, seed=42))
    assert first == list(DeterministicSampler(data, seed=42))
    assert first != list(DeterministicSampler(data, seed=43))
    next_epoch = DeterministicSampler(data, seed=42)
    next_epoch.set_epoch(1)
    assert list(next_epoch) != first
    assert list(next_epoch) == list(DeterministicSampler(data, seed=43))
    rank_zero = set(DeterministicSampler(data, seed=42, rank=0, world_size=2))
    rank_one = set(DeterministicSampler(data, seed=42, rank=1, world_size=2))
    assert rank_zero.isdisjoint(rank_one)
    resumed = list(DeterministicSampler(data, seed=42, state=SamplerState(0, 5)))
    assert resumed == first[5:]
