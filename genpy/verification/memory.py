"""CUDA memory measurement helpers."""

from __future__ import annotations

import torch


def reset_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cuda_memory_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"available": False, "peak_allocated_mib": None, "peak_reserved_mib": None}
    return {
        "available": True,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
