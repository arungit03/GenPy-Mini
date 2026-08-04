"""Allocation-free exact parameter accounting for GenPy profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from genpy.model.config import ModelConfig


@dataclass(frozen=True, slots=True)
class ParameterAudit:
    """Exact unique parameter counts and component breakdown."""

    total_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int
    token_embedding_parameters: int
    attention_parameters: int
    mlp_parameters: int
    normalization_parameters: int
    output_head_parameters: int
    per_block_parameters: int
    tied_output_weights: bool
    memory_bytes: dict[str, int]
    configuration_hash: str
    tokenizer_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible audit data."""
        return asdict(self)


def count_parameters(config: ModelConfig) -> ParameterAudit:
    """Count unique tensors from architecture dimensions without model allocation."""
    dimension = config.hidden_size
    embedding = config.vocab_size * dimension
    attention_per_block = 4 * dimension * dimension
    mlp_per_block = 3 * dimension * config.intermediate_size
    norm_per_block = 2 * dimension
    final_norm = dimension
    attention = config.num_layers * attention_per_block
    mlp = config.num_layers * mlp_per_block
    normalization = config.num_layers * norm_per_block + final_norm
    output = 0 if config.tie_word_embeddings else embedding
    total = embedding + attention + mlp + normalization + output
    return ParameterAudit(
        total_parameters=total,
        trainable_parameters=total,
        non_trainable_parameters=0,
        token_embedding_parameters=embedding,
        attention_parameters=attention,
        mlp_parameters=mlp,
        normalization_parameters=normalization,
        output_head_parameters=output,
        per_block_parameters=attention_per_block + mlp_per_block + norm_per_block,
        tied_output_weights=config.tie_word_embeddings,
        memory_bytes={"float32": total * 4, "float16": total * 2, "bfloat16": total * 2},
        configuration_hash=config.config_hash,
        tokenizer_fingerprint=str(config.tokenizer["fingerprint"]),
    )


def validate_declared_tier(config: ModelConfig, audit: ParameterAudit) -> None:
    """Reject counts grossly inconsistent with the declared model name."""
    ranges = {
        "GenPy-5M": (4_000_000, 7_000_000),
        "GenPy-25M": (15_000_000, 30_000_000),
        "GenPy-100M": (90_000_000, 110_000_000),
        "GenPy-Smoke": (1, 2_000_000),
    }
    if config.name not in ranges:
        raise ValueError(f"unknown declared model tier: {config.name}")
    lower, upper = ranges[config.name]
    if not lower <= audit.total_parameters <= upper:
        raise ValueError("exact parameter count is outside the declared tier range")
