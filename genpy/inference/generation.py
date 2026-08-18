"""Production text generation for GenPy causal language models."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Optional

import torch


_PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class GenerationConfig:
    """Controls autoregressive decoding without changing the model."""

    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None
    forbidden_token_ids: tuple[int, ...] = ()
    greedy: bool = False
    seed: Optional[int] = None
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.precision not in _PRECISION_DTYPES:
            raise ValueError(f"unsupported precision: {self.precision}")
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative")
        if any(token_id < 0 for token_id in self.forbidden_token_ids):
            raise ValueError("forbidden_token_ids must be non-negative")


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    # CUDA supports both requested low-precision modes. CPU BF16 is also
    # supported by modern PyTorch; CPU FP16 is left in the model's native
    # dtype because many CPU kernels do not implement FP16 autocast.
    if device.type == "cuda" or (device.type == "cpu" and precision == "bf16"):
        return torch.autocast(device_type=device.type, dtype=_PRECISION_DTYPES[precision])
    return nullcontext()


def _apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> None:
    if penalty == 1.0:
        return
    for batch_index in range(logits.shape[0]):
        token_ids = torch.unique(input_ids[batch_index])
        values = logits[batch_index, token_ids]
        logits[batch_index, token_ids] = torch.where(values < 0, values * penalty, values / penalty)


def _filter_logits(logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
    if config.top_k:
        k = min(config.top_k, logits.shape[-1])
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_logits.float(), dim=-1), dim=-1)
        remove = cumulative > config.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)
    return logits


def _next_token(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    logits = logits.float()
    _apply_repetition_penalty(logits, input_ids, config.repetition_penalty)
    for token_id in config.forbidden_token_ids:
        if token_id >= logits.shape[-1]:
            raise ValueError(f"forbidden token ID {token_id} is outside the model vocabulary")
        logits[:, token_id] = float("-inf")
    if config.greedy:
        return torch.argmax(logits, dim=-1)
    logits = logits / config.temperature
    logits = _filter_logits(logits, config)
    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("generation filters produced non-finite probabilities")
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig | None = None,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Generate tokens and return the prompt followed by generated tokens."""
    config = config or GenerationConfig()
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must be a rank-2 tensor [batch, sequence]")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one prompt token")
    model_device = next(model.parameters()).device
    target_device = torch.device(device) if device is not None else model_device
    if target_device != model_device:
        raise ValueError(f"model is on {model_device}, but generation requested {target_device}")
    max_seq_len = int(model.config.max_seq_len)
    if max_seq_len <= 0:
        raise ValueError("model.config.max_seq_len must be positive")
    if config.eos_token_id is not None and config.eos_token_id >= model.config.vocab_size:
        raise ValueError("eos_token_id is outside the model vocabulary")

    generated = input_ids.to(device=target_device, dtype=torch.long).clone()
    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=target_device).manual_seed(config.seed)
    was_training = model.training
    model.eval()
    finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=target_device)
    try:
        with torch.inference_mode(), _autocast_context(target_device, config.precision):
            for _ in range(config.max_new_tokens):
                context = generated[:, -max_seq_len:]
                logits = model(context)[:, -1, :]
                next_tokens = _next_token(logits, generated, config, generator)
                if config.eos_token_id is not None:
                    if finished.any():
                        next_tokens = torch.where(
                            finished,
                            torch.full_like(next_tokens, config.eos_token_id),
                            next_tokens,
                        )
                    finished |= next_tokens == config.eos_token_id
                generated = torch.cat((generated, next_tokens[:, None]), dim=1)
                if config.eos_token_id is not None and bool(finished.all()):
                    break
    finally:
        if was_training:
            model.train()
    return generated


__all__ = ["GenerationConfig", "generate"]
