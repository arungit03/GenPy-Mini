import torch

from genpy.training.scheduler import WarmupCosineScheduler


def test_warmup_cosine_and_state_restore() -> None:
    parameter = torch.nn.Parameter(torch.ones(1)); optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = WarmupCosineScheduler(optimizer, 0.1, 0.01, 2, 10)
    values = [scheduler.get_last_lr()[0]]
    for _ in range(10): scheduler.step(); values.append(scheduler.get_last_lr()[0])
    assert values[0] == 0.0 and values[2] == 0.1 and values[-1] == 0.01
    other = WarmupCosineScheduler(optimizer, 0.1, 0.01, 2, 10); other.load_state_dict(scheduler.state_dict())
    assert other.get_last_lr() == scheduler.get_last_lr()
