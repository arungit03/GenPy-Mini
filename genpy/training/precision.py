"""Device-aware autocast and gradient-scaling policy."""

from contextlib import nullcontext

import torch


class PrecisionManager:
    def __init__(self, requested: str = "auto", device: torch.device | str = "cpu") -> None:
        if requested not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("precision must be auto, fp32, fp16, or bf16")
        self.device = torch.device(device)
        if requested == "auto":
            self.mode = "bf16" if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
        elif self.device.type != "cuda" and requested != "fp32":
            raise ValueError(f"precision '{requested}' requires CUDA; CPU supports fp32")
        elif requested == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 is not supported by this CUDA device")
        else:
            self.mode = requested
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mode == "fp16" and self.device.type == "cuda")

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def autocast(self):
        if self.mode == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if self.mode == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler.is_enabled():
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler.is_enabled():
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def state_dict(self) -> dict:
        return {"requested_mode": self.mode, "scaler": self.scaler.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if state.get("requested_mode") != self.mode:
            raise ValueError("precision mode mismatch")
        self.scaler.load_state_dict(state.get("scaler", {}))
