"""Evaluate a GenPy checkpoint on a packed validation token file."""

import argparse
import json
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import torch

from genpy.config import load_model_config
from genpy.evaluation import evaluate_token_file
from genpy.inference import load_checkpoint_weights
from genpy.model import GenPyForCausalLM


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sequence_length <= 0:
        raise ValueError("sequence-length must be positive")

    model_config = load_model_config(args.model_config)
    device = _device(args.device)
    model = GenPyForCausalLM(model_config)
    payload = load_checkpoint_weights(model, args.checkpoint, model_config)
    model.to(device)
    if args.precision != "fp32":
        model.to(dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[args.precision])
    result = evaluate_token_file(
        model,
        args.validation_data,
        args.sequence_length,
        metadata_path=args.validation_metadata,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    state = payload.get("training_state") or {}
    output = {
        "model": model_config.name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_step": int(state.get("global_step", 0)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "sequence_length": result.sequence_length,
        **result.to_dict(),
    }
    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
