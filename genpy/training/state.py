"""Versioned resumable trainer state."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TrainingState:
    schema_version: int = 1
    run_id: str = "genpy200m_pretrain_v1"
    global_step: int = 0
    micro_step: int = 0
    tokens_seen: int = 0
    windows_seen: int = 0
    best_validation_loss: float | None = None
    last_validation_loss: float | None = None
    current_learning_rate: float = 0.0
    elapsed_training_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "TrainingState":
        return cls(**value)
