"""Optional CUDA smoke test for the full GenPy model."""

import argparse
import sys
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import torch

from genpy.config import load_model_config
from genpy.model import GenPyForCausalLM
from genpy.verification import causal_lm_loss, gradient_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA unavailable; GPU smoke test skipped gracefully.")
        return 0
    if args.sequence_length <= 0:
        print("sequence length must be positive", file=sys.stderr)
        return 2
    config = load_model_config(Path(__file__).resolve().parents[1] / "configs" / "model_200m.yaml")
    if args.sequence_length > config.max_seq_len:
        print(f"sequence length cannot exceed {config.max_seq_len}", file=sys.stderr)
        return 2
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    if args.precision == "bf16" and not bf16_supported:
        print("BF16 is not supported on this CUDA device; smoke test skipped gracefully.")
        return 0
    device = torch.device("cuda")
    model = GenPyForCausalLM(config).to(device)
    input_ids = torch.randint(0, config.vocab_size, (1, args.sequence_length), device=device)
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else torch.autocast("cuda", enabled=False)
    with autocast:
        logits = model(input_ids)
        loss = causal_lm_loss(logits, input_ids)
    if not bool(torch.isfinite(logits).all().item()) or not bool(torch.isfinite(loss).item()):
        print("non-finite logits or loss", file=sys.stderr)
        return 1
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"BF16 supported: {bf16_supported}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Loss: {loss.item():.6f}")
    if args.backward:
        loss.backward()
        report = gradient_report(model)
        print(f"Missing gradients: {len(report['missing'])}")
        print(f"Non-finite gradients: {len(report['non_finite'])}")
        if report["missing"] or report["non_finite"]:
            return 1
    print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated(device) / (1024 ** 3):.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
