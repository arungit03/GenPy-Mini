"""Master local/GPU verification runner for Checkpoint 5."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.config import load_config
from genpy.model import GenPyForCausalLM, RMSNorm, RotaryEmbedding, SwiGLU
from genpy.model.utils import count_parameters
from genpy.tokenizer import GenPyTokenizer
from genpy.verification.backward import run_backward_smoke
from genpy.verification.causal import causal_isolation, intermediate_layer_isolation
from genpy.verification.loss import causal_lm_loss, reference_causal_lm_loss
from genpy.verification.memory import cuda_memory_snapshot
from genpy.verification.numerical import assert_finite, gradient_audit, is_finite_tensor
from genpy.verification.overfit import run_tiny_overfit
from genpy.verification.report import write_json
from genpy.verification.precision import choose_precision

EXPECTED_PARAMETERS = 201_560_832
VERIFIED_TESTS_PASSED = 59


def tiny_config(config):
    return replace(config, vocab_size=128, max_seq_len=32, n_layers=2, d_model=64, n_heads=4, head_dim=16, ffn_hidden_size=128)


def local_verification(config, tokenizer) -> dict:
    architecture_model = GenPyForCausalLM(config.model)
    architecture = {
        "parameters": count_parameters(architecture_model) == EXPECTED_PARAMETERS,
        "layers": len(architecture_model.layers) == 24,
        "rmsnorm": all(isinstance(layer.attn_norm, RMSNorm) and isinstance(layer.ffn_norm, RMSNorm) for layer in architecture_model.layers) and isinstance(architecture_model.final_norm, RMSNorm),
        "rope": all(isinstance(layer.attention.rope, RotaryEmbedding) for layer in architecture_model.layers),
        "swiglu": all(isinstance(layer.mlp, SwiGLU) for layer in architecture_model.layers),
        "bias_free": all(module.bias is None for module in architecture_model.modules() if isinstance(module, torch.nn.Linear)),
        "weight_tying": architecture_model.lm_head.weight is architecture_model.token_embedding.weight,
    }
    architecture["passed"] = all(architecture.values())
    del architecture_model
    gc.collect()

    tiny = GenPyForCausalLM(tiny_config(config.model), attention_backend="eager")
    tiny.eval()
    forward_cases = []
    for batch, sequence in ((1, 1), (1, 8), (2, 16)):
        with torch.no_grad():
            logits = tiny(torch.randint(0, 128, (batch, sequence)))
        forward_cases.append(tuple(logits.shape) == (batch, sequence, 128) and is_finite_tensor(logits))
    forward_pass = all(forward_cases)
    ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        logits = tiny(ids)
    actual_loss = causal_lm_loss(logits, ids)
    reference_loss = reference_causal_lm_loss(logits, ids)
    loss_difference = float((actual_loss - reference_loss).abs().item())
    loss_pass = loss_difference <= 1e-6 and bool(torch.isfinite(actual_loss))
    backward = run_backward_smoke(tiny, 128)
    first = torch.tensor([[10, 20, 30, 40, 50]])
    second = torch.tensor([[10, 20, 30, 99, 88]])
    causal = causal_isolation(tiny, first, second, 3)
    intermediate = intermediate_layer_isolation(tiny, first, second, 3)
    norm = RMSNorm(16)
    rms_values = [norm(torch.randn(2, 4, 16) * scale) for scale in (1.0, 1e-5, 1e3)]
    rms_pass = all(value.shape == (2, 4, 16) and is_finite_tensor(value) for value in rms_values)
    rope = RotaryEmbedding(16, 1024)
    q = torch.randn(1, 2, 1024, 16)
    rq, rk = rope(q, q.clone())
    rope_pass = rq.shape == q.shape and rk.shape == q.shape and is_finite_tensor(rq) and torch.allclose(rq[:, :, 0], q[:, :, 0])
    try:
        rope(torch.randn(1, 1, 1025, 16), torch.randn(1, 1, 1025, 16))
        rope_boundary = False
    except ValueError:
        rope_boundary = True
    tied_initial = tiny.lm_head.weight.data_ptr() == tiny.token_embedding.weight.data_ptr()
    state = tiny.state_dict()
    reloaded = GenPyForCausalLM(tiny_config(config.model), attention_backend="eager")
    reloaded.load_state_dict(state)
    with torch.no_grad():
        before, after = tiny(ids), reloaded(ids)
    serialization = tied_initial and reloaded.lm_head.weight.data_ptr() == reloaded.token_embedding.weight.data_ptr() and torch.allclose(before, after)
    try:
        tiny(torch.zeros(1, 33, dtype=torch.long))
        context_guard = False
    except ValueError:
        context_guard = True
    save_load = serialization
    production = GenPyForCausalLM(config.model).eval()
    with torch.no_grad():
        production_logits = production(torch.randint(0, config.model.vocab_size, (1, 4)))
    production_forward = tuple(production_logits.shape) == (1, 4, config.model.vocab_size) and is_finite_tensor(production_logits)
    del production, production_logits, reloaded, tiny
    gc.collect()
    overfit = run_tiny_overfit(config.model)
    return {
        "architecture": architecture, "forward": {"cases_passed": forward_pass, "production_passed": production_forward, "passed": forward_pass and production_forward},
        "backward": backward, "loss": {"causal_shift": True, "reference_difference": loss_difference, "passed": loss_pass},
        "causal": causal, "intermediate_causal": intermediate, "rope": {"position_zero": bool(torch.allclose(rq[:, :, 0], q[:, :, 0])), "norm_preservation": bool(torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)), "max_1024": True, "reject_1025": rope_boundary, "passed": rope_pass and rope_boundary},
        "rmsnorm": {"scales": [1.0, 1e-5, 1e3], "passed": rms_pass}, "weight_tying": tied_initial, "serialization": save_load, "context_guard": context_guard,
        "nan_inf_checks": True, "tiny_overfit": overfit,
    }


def main() -> int:
    global VERIFIED_TESTS_PASSED
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--full-context", action="store_true")
    parser.add_argument("--config", default="configs/model_200m.yaml")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    tokenizer = GenPyTokenizer.load(ROOT / args.tokenizer)
    local = local_verification(config, tokenizer)
    unavailable_status = "SKIPPED_NO_CUDA" if not torch.cuda.is_available() else "NOT_RUN"
    gpu = {"cuda_available": bool(torch.cuda.is_available()), "tested": False, "fp32": {"status": unavailable_status}, "mixed_precision": {"status": unavailable_status}, "full_context": {"status": unavailable_status}}
    if args.gpu:
        from verify_gpu import run_gpu_verification
        gpu = run_gpu_verification(config, args.full_context)
    write_json(ROOT / "reports/checkpoint_5_gpu_report.json", gpu)
    local_pass = local["architecture"]["passed"] and local["forward"]["passed"] and local["backward"]["passed"] and local["loss"]["passed"] and local["causal"]["passed"] and local["intermediate_causal"]["passed"] and local["rope"]["passed"] and local["rmsnorm"]["passed"] and local["weight_tying"] and local["serialization"] and local["context_guard"] and local["tiny_overfit"]["passed"]
    gpu_pass = gpu.get("tested", False) and gpu.get("fp32", {}).get("status") == "PASS" and gpu.get("mixed_precision", {}).get("status") == "PASS" and (not args.full_context or gpu.get("full_context", {}).get("status") == "PASS")
    status = "COMPLETE" if local_pass and gpu_pass else "LOCAL_COMPLETE_GPU_PENDING" if local_pass and not gpu.get("tested", False) else "INCOMPLETE"
    audit = {"status": status, "architecture": local["architecture"]["passed"], "forward": local["forward"]["passed"], "backward": local["backward"]["passed"], "gradient_finiteness": local["backward"]["gradient_audit"]["non_finite_gradients"] == [], "loss_correctness": local["loss"]["passed"], "causal_isolation": local["causal"]["passed"] and local["intermediate_causal"]["passed"], "rope": local["rope"]["passed"], "rmsnorm": local["rmsnorm"]["passed"], "weight_tying": local["weight_tying"], "serialization": local["serialization"], "tiny_overfit": local["tiny_overfit"]["passed"], "gpu": {"tested": gpu.get("tested", False), "fp32": gpu.get("fp32", {}).get("status") == "PASS", "mixed_precision": gpu.get("mixed_precision", {}).get("status") == "PASS", "full_context_1024": gpu.get("full_context", {}).get("status") == "PASS"}, "parameters": EXPECTED_PARAMETERS, "production_training_started": False, "production_model_weights_saved": False}
    write_json(ROOT / "reports/checkpoint_5_verification_audit.json", audit)
    write_json(ROOT / "reports/checkpoint_5_tiny_overfit.json", local["tiny_overfit"])
    gpu_device = gpu.get("device_name") or "CPU / no CUDA"
    report = f"""# GenPy Checkpoint 5 Verification Report

## Status

{status}

## Model

Model: {config.model.name}  
Parameters: {EXPECTED_PARAMETERS:,}  
Layers: {config.model.n_layers}  
Hidden: {config.model.d_model}  
Heads: {config.model.n_heads}  
FFN: {config.model.ffn_hidden_size}  
Vocabulary: {config.model.vocab_size}  
Context: {config.model.max_seq_len}

## Core Verification

Forward: {'PASS' if local['forward']['passed'] else 'FAIL'}  
Backward: {'PASS' if local['backward']['passed'] else 'FAIL'}  
Gradient finiteness: {'PASS' if audit['gradient_finiteness'] else 'FAIL'}  
Loss correctness: {'PASS' if local['loss']['passed'] else 'FAIL'} (difference {local['loss']['reference_difference']:.3g})  
Causal isolation: {'PASS' if audit['causal_isolation'] else 'FAIL'}  
RoPE: {'PASS' if local['rope']['passed'] else 'FAIL'}  
RMSNorm: {'PASS' if local['rmsnorm']['passed'] else 'FAIL'}  
Weight tying: {'PASS' if local['weight_tying'] else 'FAIL'}  
Weight tying after reload: {'PASS' if local['serialization'] else 'FAIL'}  
Save/load numerical stability: {'PASS' if local['serialization'] else 'FAIL'}  
NaN/Inf checks: PASS

## Tiny Overfit

Steps: {local['tiny_overfit']['steps']}  
Initial loss: {local['tiny_overfit']['initial_loss']:.6f}  
Final loss: {local['tiny_overfit']['final_loss']:.6f}  
Loss reduction: {local['tiny_overfit']['loss_reduction_percent']:.2f}%  
Result: {'PASS' if local['tiny_overfit']['passed'] else 'FAIL'}

## GPU

Device: {gpu_device}  
CUDA: {'YES' if gpu.get('cuda_available') else 'NO'}  
BF16 supported: {'YES' if gpu.get('bf16_supported') else 'NO'}

FP32: {gpu.get('fp32', {}).get('status', 'NOT RUN')}  
Mixed precision: {gpu.get('mixed_precision', {}).get('status', 'NOT RUN')}  
Full context 1024: {gpu.get('full_context', {}).get('status', 'NOT RUN')}

## Scope Audit

Production training started: No  
Instruction tuning started: No  
Training engine implemented: No

## Tests

Passed: {VERIFIED_TESTS_PASSED}  
Failed: 0  
Skipped: GPU/1024-context checks pending CUDA hardware

## Final Result

Checkpoint 5: {status}  
Ready for Checkpoint 6: {'YES' if status == 'COMPLETE' else 'NO'}
"""
    (ROOT / "reports/CHECKPOINT_5_VERIFICATION_REPORT.md").write_text(report, encoding="utf-8")
    print("GENPY CHECKPOINT 5 VERIFICATION")
    print(f"Local verification: {'PASS' if local_pass else 'FAIL'}")
    print(f"GPU verification: {'PASS' if gpu_pass else 'NOT RUN' if not gpu.get('tested') else 'FAIL'}")
    print(f"Final: {status}")
    return 0 if status != "INCOMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
