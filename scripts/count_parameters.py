"""Report theoretical and actual GenPy parameter counts."""

from pathlib import Path

try:
    from genpy.config import load_model_config
    from genpy.model import GenPyForCausalLM
except ModuleNotFoundError:  # Allows direct execution from the scripts directory.
    from _bootstrap import ensure_project_root

    ensure_project_root()
    from genpy.config import load_model_config
    from genpy.model import GenPyForCausalLM


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
    config_path = Path(__file__).resolve().parents[1] / "configs" / "model_200m.yaml"
    config = load_model_config(config_path)
    estimate = theoretical_parameter_estimate()
    model = GenPyForCausalLM(config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    difference = trainable - estimate
    print(f"Model: {config.name}")
    print(f"Theoretical parameter estimate: {estimate:,}")
    print(f"Actual trainable parameters: {trainable:,}")
    print(f"Actual total parameters: {total:,}")
    print(f"Difference: {difference:,}")
    print("Parameter breakdown (unique trainable parameters):")
    for name, count in model.parameter_breakdown().items():
        print(f"  {name}: {count:,}")
    print("Weights-only memory estimate:")
    print(f"  FP32: {trainable * 4 / (1024 ** 2):.2f} MiB")
    print(f"  FP16/BF16: {trainable * 2 / (1024 ** 2):.2f} MiB")
    print("These estimates exclude gradients, optimizer states, and activations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
