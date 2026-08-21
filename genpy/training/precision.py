"""Training-engine precision selection and scaler handling."""

from __future__ import annotations

from contextlib import nullcontext

import torch


class PrecisionManager:
    def __init__(self, requested: str, device: torch.device) -> None:
        requested = requested.lower()
        if requested not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("precision must be auto, fp32, bf16, or fp16")
        if device.type != "cuda":
            if requested in {"bf16", "fp16"}:
                raise ValueError(f"{requested} requires CUDA in the training engine")
            self.mode, self.dtype = "fp32", torch.float32
        elif requested == "auto":
            self.mode, self.dtype = (("bf16", torch.bfloat16) if torch.cuda.is_bf16_supported() else ("fp16", torch.float16))
        elif requested == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 is not supported by this CUDA device")
            self.mode, self.dtype = "bf16", torch.bfloat16
        else:
            self.mode, self.dtype = requested, torch.float16
        self.scaler = None
        if self.mode == "fp16":
            try:
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            except (AttributeError, TypeError):
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def autocast(self):
        if self.mode in {"bf16", "fp16"}:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    def backward(self, loss) -> None:
        self.scaler.scale(loss).backward() if self.scaler else loss.backward()

    def unscale(self, optimizer) -> None:
        if self.scaler:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer) -> None:
        if self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def state_dict(self) -> dict:
        return self.scaler.state_dict() if self.scaler else {"enabled": False}

    def load_state_dict(self, state: dict) -> None:
        if self.scaler and state.get("enabled", True):
            self.scaler.load_state_dict(state)
