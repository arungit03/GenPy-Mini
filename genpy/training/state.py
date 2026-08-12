"""Serializable counters for optimizer and microbatch progress."""

from dataclasses import asdict, dataclass
import time


@dataclass
class TrainingState:
    global_step: int = 0
    micro_step: int = 0
    tokens_seen: int = 0
    samples_seen: int = 0
    optimizer_steps: int = 0
    best_validation_loss: float | None = None
    elapsed_seconds: float = 0.0

    def state_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict) -> "TrainingState":
        return cls(**state)

    def begin_timer(self) -> float:
        return time.perf_counter()
