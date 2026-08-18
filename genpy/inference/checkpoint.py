"""Inference-only loading of existing GenPy checkpoint model weights."""

from dataclasses import asdict
from pathlib import Path

import torch


def load_checkpoint_weights(model: torch.nn.Module, checkpoint: Path | str, model_config=None) -> dict:
    """Load only ``payload['model']`` with strict state-dict validation."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("checkpoint payload does not contain model weights")
    if model_config is not None and payload.get("model_config") != asdict(model_config):
        raise ValueError("checkpoint model configuration is incompatible")
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("checkpoint model weights are incompatible") from error
    return payload


__all__ = ["load_checkpoint_weights"]
