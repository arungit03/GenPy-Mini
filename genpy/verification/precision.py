"""Verification-only precision selection and autocast helpers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrecisionChoice:
    mode: str
    dtype: torch.dtype
    supported: bool
    reason: str


def choose_precision(mode: str = "auto", device: torch.device | str | None = None) -> PrecisionChoice:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    requested = mode.lower()
    if requested == "fp32":
        return PrecisionChoice("fp32", torch.float32, True, "explicit FP32")
    if device.type != "cuda":
        return PrecisionChoice("fp32", torch.float32, requested == "auto" or requested == "fp32", "CPU verification uses FP32")
    bf16 = bool(torch.cuda.is_bf16_supported())
    if requested == "bf16":
        return PrecisionChoice("bf16", torch.bfloat16, bf16, "CUDA BF16 capability")
    if requested == "fp16":
        return PrecisionChoice("fp16", torch.float16, True, "CUDA FP16 capability")
    if requested == "auto":
        return PrecisionChoice("bf16" if bf16 else "fp16", torch.bfloat16 if bf16 else torch.float16, True, "automatic CUDA selection")
    raise ValueError("precision mode must be auto, fp32, bf16, or fp16")


def autocast_context(choice: PrecisionChoice, device: torch.device | str):
    device = torch.device(device)
    if device.type == "cuda" and choice.mode in {"bf16", "fp16"}:
        return torch.autocast(device_type="cuda", dtype=choice.dtype)
    return nullcontext()
