"""CUDA production-model verification; skips cleanly on CPU-only machines."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.config import load_config
from genpy.model import GenPyForCausalLM
from genpy.verification.loss import causal_lm_loss
from genpy.verification.memory import cuda_memory_snapshot, reset_cuda_memory
from genpy.verification.numerical import gradient_audit
from genpy.verification.precision import autocast_context, choose_precision


def _run_mode(config, mode: str, sequence_length: int = 32) -> dict:
    device = torch.device("cuda")
    choice = choose_precision(mode, device)
    if not choice.supported:
        return {"status": "SKIPPED_UNSUPPORTED", "mode": choice.mode, "reason": choice.reason}
    reset_cuda_memory()
    model = None
    try:
        model = GenPyForCausalLM(config.model).to(device)
        model.train()
        ids = torch.randint(0, config.model.vocab_size, (1, sequence_length), device=device)
        start = time.perf_counter()
        with autocast_context(choice, device):
            logits = model(ids)
            loss = causal_lm_loss(logits.float(), ids)
        forward_seconds = time.perf_counter() - start
        finite_forward = bool(torch.isfinite(logits).all() and torch.isfinite(loss))
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_seconds = time.perf_counter() - start
        audit = gradient_audit(model)
        result = {
            "status": "PASS" if finite_forward and not audit.non_finite_gradients else "FAIL",
            "mode": choice.mode, "sequence_length": sequence_length, "loss": float(loss.detach().item()),
            "forward": "PASS" if finite_forward else "FAIL", "backward": "PASS" if not audit.non_finite_gradients else "FAIL",
            "gradient_audit": audit.to_dict(), "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
            **cuda_memory_snapshot(),
        }
        return result
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower()
        return {"status": "FAIL_OOM" if is_oom else "FAIL", "mode": choice.mode, "sequence_length": sequence_length, "error": str(exc)}
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run_gpu_verification(config, full_context: bool = False) -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "tested": False, "device_name": None, "bf16_supported": False, "fp32": {"status": "SKIPPED_NO_CUDA"}, "mixed_precision": {"status": "SKIPPED_NO_CUDA"}, "full_context": {"status": "SKIPPED_NO_CUDA"}}
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    fp32 = _run_mode(config, "fp32", 32)
    mixed_mode = "bf16" if bf16_supported else "fp16"
    mixed = _run_mode(config, mixed_mode, 32)
    full = {"status": "NOT_RUN"}
    if full_context:
        full = _run_mode(config, "bf16" if bf16_supported else "fp32", 1024)
    return {"cuda_available": True, "tested": True, "device_name": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda, "pytorch_version": torch.__version__, "bf16_supported": bf16_supported, "fp32": fp32, "mixed_precision": mixed, "full_context": full}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_200m.yaml")
    parser.add_argument("--full-context", action="store_true")
    args = parser.parse_args()
    result = run_gpu_verification(load_config(ROOT / args.config), args.full_context)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/checkpoint_5_gpu_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    failed = any(result.get(key, {}).get("status", "").startswith("FAIL") for key in ("fp32", "mixed_precision", "full_context"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
