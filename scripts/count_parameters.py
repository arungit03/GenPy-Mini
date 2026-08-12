"""Print a theoretical GenPy parameter estimate; no model is instantiated."""

from pathlib import Path

try:
    from genpy.config import load_model_config
except ModuleNotFoundError:  # Allows direct execution from the scripts directory.
    from types import SimpleNamespace

    import yaml

    def load_model_config(path):
        with open(path, encoding="utf-8") as handle:
            return SimpleNamespace(**yaml.safe_load(handle)["model"])


def theoretical_parameter_estimate() -> int:
    """Estimate parameters using tied embeddings and a standard Transformer budget."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "model_200m.yaml"
    config = load_model_config(config_path)
    embedding = config.vocab_size * config.hidden_size
    attention_per_layer = 4 * config.hidden_size * config.hidden_size
    swiglu_per_layer = 3 * config.hidden_size * config.intermediate_size
    norms_per_layer = 2 * config.hidden_size
    final_norm = config.hidden_size
    return embedding + config.num_layers * (
        attention_per_layer + swiglu_per_layer + norms_per_layer
    ) + final_norm


def main() -> int:
    estimate = theoretical_parameter_estimate()
    print("Theoretical parameter estimate")
    print("Assumptions: tied input/output embeddings; four dense attention projections;"
          " three dense SwiGLU projections; two RMSNorm scale vectors per layer;")
    print(f"Estimated parameters: {estimate:,}")
    print("Exact parameter counting will become available after the GenPy model is implemented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
