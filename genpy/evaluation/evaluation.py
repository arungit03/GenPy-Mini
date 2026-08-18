"""Token-weighted causal-language-model evaluation."""

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from genpy.training.data import PackedTokenDataset


_PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class EvaluationResult:
    validation_loss: float
    perplexity: float
    evaluated_tokens: int
    evaluation_windows: int
    sequence_length: int

    def to_dict(self) -> dict[str, object]:
        return {
            "validation_loss": self.validation_loss,
            "perplexity": self.perplexity,
            "evaluated_tokens": self.evaluated_tokens,
            "evaluation_windows": self.evaluation_windows,
            "sequence_length": self.sequence_length,
        }


def _autocast_context(device: torch.device, precision: str):
    if precision not in _PRECISION_DTYPES:
        raise ValueError(f"unsupported precision: {precision}")
    if precision == "fp32":
        return nullcontext()
    if device.type == "cuda" or (device.type == "cpu" and precision == "bf16"):
        return torch.autocast(device_type=device.type, dtype=_PRECISION_DTYPES[precision])
    return nullcontext()


def _normalize_device(device: str | torch.device) -> torch.device:
    """Resolve an unspecified CUDA index using PyTorch's current device."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device("cuda", index)
    return resolved


def _devices_match(model_device: torch.device, requested_device: str | torch.device) -> bool:
    """Compare concrete devices while treating ``cuda`` as the current CUDA device."""
    return _normalize_device(model_device) == _normalize_device(requested_device)


def evaluate_packed_dataset(
    model: torch.nn.Module,
    dataset: PackedTokenDataset,
    *,
    device: str | torch.device = "cpu",
    precision: str = "fp32",
    batch_size: int = 1,
) -> EvaluationResult:
    """Evaluate complete packed windows with exact token-weighted NLL."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(dataset) == 0:
        raise ValueError(
            "validation dataset contains fewer than sequence_length + 1 tokens; no complete windows"
        )
    target_device = _normalize_device(device)
    model_device = next(model.parameters()).device
    if not _devices_match(model_device, target_device):
        raise ValueError(f"model is on {model_device}, but evaluation requested {target_device}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    total_nll = 0.0
    evaluated_tokens = 0
    evaluation_windows = 0
    try:
        with torch.inference_mode(), _autocast_context(target_device, precision):
            for batch in loader:
                input_ids = batch["input_ids"].to(target_device)
                targets = batch["targets"].to(target_device)
                logits = model(input_ids).float()
                total_nll += float(
                    F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                        reduction="sum",
                    ).item()
                )
                evaluated_tokens += int(targets.numel())
                evaluation_windows += int(input_ids.shape[0])
    finally:
        if was_training:
            model.train()
    loss = total_nll / evaluated_tokens
    return EvaluationResult(
        validation_loss=loss,
        perplexity=math.exp(loss),
        evaluated_tokens=evaluated_tokens,
        evaluation_windows=evaluation_windows,
        sequence_length=dataset.sequence_length,
    )


def evaluate_token_file(
    model: torch.nn.Module,
    token_path: Path | str,
    sequence_length: int,
    *,
    metadata_path: Path | str | None = None,
    device: str | torch.device = "cpu",
    precision: str = "fp32",
    batch_size: int = 1,
) -> EvaluationResult:
    """Construct the existing packed dataset and evaluate it."""
    dataset = PackedTokenDataset(Path(token_path), sequence_length, metadata_path)
    try:
        return evaluate_packed_dataset(
            model,
            dataset,
            device=device,
            precision=precision,
            batch_size=batch_size,
        )
    finally:
        dataset.close()


__all__ = ["EvaluationResult", "evaluate_packed_dataset", "evaluate_token_file"]
