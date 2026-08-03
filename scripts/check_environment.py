"""Report the local GenPy development environment."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

REQUIRED_DIRECTORIES = (
    "configs",
    "configs/model",
    "data/raw",
    "data/cleaned",
    "data/tokenized",
    "data/instruction",
    "data/evaluation",
    "docs",
    "scripts",
    "src/genpy",
    "tests",
    "artifacts",
    "checkpoints",
    "logs",
)


def project_root() -> Path:
    """Return the project root based on this script location."""
    return Path(__file__).resolve().parents[1]


def get_torch_report() -> list[str]:
    """Return PyTorch and CUDA status lines without requiring CUDA."""
    try:
        import torch
    except ImportError:
        return ["PyTorch: not installed", "CUDA: unavailable", "Mode: CPU-only"]

    lines = [f"PyTorch: {torch.__version__}"]
    cuda_available = torch.cuda.is_available()
    lines.append(f"CUDA available: {cuda_available}")

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        lines.append(f"CUDA device: {device_name}")
    else:
        lines.append("Mode: CPU-only")

    return lines


def get_directory_report(root: Path) -> list[str]:
    """Return existence checks for required project directories."""
    lines = ["Required directories:"]
    for directory in REQUIRED_DIRECTORIES:
        status = "ok" if (root / directory).is_dir() else "missing"
        lines.append(f"  {directory}: {status}")
    return lines


def build_report() -> str:
    """Build a human-readable environment report."""
    root = project_root()
    lines = [
        "GenPy environment report",
        f"Project root: {root}",
        f"Python: {sys.version.split()[0]}",
        f"Executable: {sys.executable}",
        f"System: {platform.system()} {platform.release()}",
        f"Machine: {platform.machine()}",
        f"Processor: {platform.processor() or 'unknown'}",
    ]
    lines.extend(get_torch_report())
    lines.extend(get_directory_report(root))
    return "\n".join(lines)


def main() -> None:
    """Print the environment report."""
    print(build_report())


if __name__ == "__main__":
    main()
