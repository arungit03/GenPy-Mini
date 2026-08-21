"""Compute-device selection and environment reporting."""

from typing import Any

import torch


def get_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, and finally CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _bf16_supported() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        return False


def get_device_info() -> dict[str, Any]:
    """Return safe, structured information about the local PyTorch runtime."""
    cuda_available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if cuda_available else 0
    gpu_name = None
    if cuda_available and count:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except RuntimeError:
            gpu_name = None
    mps = getattr(torch.backends, "mps", None)
    mps_available = bool(mps is not None and mps.is_available())
    return {
        "device": str(get_device()),
        "device_type": get_device().type,
        "cuda_available": cuda_available,
        "cuda_device_count": count,
        "gpu_name": gpu_name,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mps_available": mps_available,
        "bf16_supported": _bf16_supported(),
    }
