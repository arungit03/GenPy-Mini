"""Conservative analytical memory estimates for later training planning."""

from __future__ import annotations

from typing import Any

from genpy.model.config import ModelConfig
from genpy.model.parameter_count import count_parameters


def estimate_training_memory(
    config: ModelConfig, sequence_length: int, micro_batch_size: int
) -> dict[str, Any]:
    """Estimate memory by dtype; activation values are explicitly approximate."""
    if not 1 <= sequence_length <= config.context_length or micro_batch_size < 1:
        raise ValueError("invalid sequence length or micro batch size")
    parameters = count_parameters(config).total_parameters
    results: dict[str, Any] = {}
    for dtype, bytes_per_value in (("float32", 4), ("float16", 2), ("bfloat16", 2)):
        parameter_bytes = parameters * bytes_per_value
        gradient_bytes = parameters * bytes_per_value
        optimizer_bytes = parameters * 8
        master_bytes = 0 if dtype == "float32" else parameters * 4
        inputs = micro_batch_size * sequence_length * 8 * 2
        activation_factor = 8 if config.gradient_checkpointing else 16
        activations = (
            micro_batch_size * sequence_length * config.hidden_size * config.num_layers
            * bytes_per_value * activation_factor
        )
        total = (
            parameter_bytes
            + gradient_bytes
            + optimizer_bytes
            + master_bytes
            + inputs
            + activations
        )
        results[dtype] = {
            "parameter_bytes": parameter_bytes,
            "gradient_bytes": gradient_bytes,
            "adamw_state_bytes": optimizer_bytes,
            "master_weight_bytes": master_bytes,
            "input_and_label_bytes": inputs,
            "approximate_activation_bytes": activations,
            "estimated_total_bytes": total,
            "future_8bit_optimizer_state_bytes": parameters * 2,
        }
    return {
        "model": config.name,
        "sequence_length": sequence_length,
        "micro_batch_size": micro_batch_size,
        "gradient_checkpointing": config.gradient_checkpointing,
        "activation_estimate_notice": "Analytical estimate only; it does not guarantee GPU fit.",
        "dtypes": results,
    }
