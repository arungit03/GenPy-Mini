"""Parameter-count helpers for future GenPy model implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_model_config(path: Path) -> dict[str, Any]:
    """Load a YAML model configuration file."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict) or "model" not in data:
        msg = f"Missing 'model' section in {path}"
        raise ValueError(msg)
    return data


def estimate_decoder_parameter_count(config: dict[str, Any]) -> int:
    """Estimate parameter count from config values until the real model exists.

    This is a planning helper only. The final count must be verified against an actual
    instantiated PyTorch model in a later phase.
    """
    model = config["model"]
    vocab_size = int(model["vocab_size"])
    hidden_size = int(model["hidden_size"])
    num_layers = int(model["num_layers"])
    intermediate_size = int(model["intermediate_size"])

    embeddings = vocab_size * hidden_size
    attention = num_layers * (4 * hidden_size * hidden_size)
    feed_forward = num_layers * (3 * hidden_size * intermediate_size)
    norms = num_layers * (2 * hidden_size)
    final_norm = hidden_size
    return embeddings + attention + feed_forward + norms + final_norm


def main() -> None:
    """Print rough parameter estimates for config files."""
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "configs" / "model").glob("*.yaml")):
        config = load_model_config(path)
        estimate = estimate_decoder_parameter_count(config)
        print(f"{path.name}: approximate planning estimate {estimate:,} parameters")


if __name__ == "__main__":
    main()
