import torch

from genpy.training.scheduler import WarmupCosineScheduler, warmup_cosine_lr


def test_warmup_cosine_properties_and_state():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = WarmupCosineScheduler(optimizer, 10, 1.0, 0.1, 0.2)
    values = [scheduler.get_last_lr()[0]]
    for _ in range(10):
        scheduler.step()
        values.append(scheduler.get_last_lr()[0])
    assert values[0] > 0
    assert max(values) <= 1.0
    assert values[-1] == 0.1
    assert warmup_cosine_lr(0, 5, 1.0, 0.1, 0) == 1.0
    state = scheduler.state_dict()
    other = WarmupCosineScheduler(torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1.0), 10, 1.0, 0.1, 0.2)
    other.load_state_dict(state)
    assert other.state_dict() == state
