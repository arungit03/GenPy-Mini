"""Run a CPU-safe sanity check for the GenPy Checkpoint 1 foundation."""

from pathlib import Path
import platform
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from genpy.config import load_config
from genpy.utils.device import get_device, get_device_info
from genpy.utils.paths import CONFIG_DIR


def main() -> int:
    config = load_config(CONFIG_DIR / "model_200m.yaml")
    info = get_device_info()
    print("GenPy Environment Check")
    print("=======================")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"CUDA available: {info['cuda_available']}")
    print(f"CUDA devices: {info['cuda_device_count']}")
    print(f"Selected device: {get_device()}")
    print(f"GPU: {info['gpu_name'] or 'None'}")
    print(f"BF16 supported: {info['bf16_supported']}")
    print()
    print("Configuration:")
    print(f"Model: {config.model.name}")
    print(f"Layers: {config.model.n_layers}")
    print(f"Hidden size: {config.model.d_model}")
    print(f"Attention heads: {config.model.n_heads}")
    print(f"Head dimension: {config.model.head_dim}")
    print(f"FFN hidden size: {config.model.ffn_hidden_size}")
    print(f"Vocabulary: {config.model.vocab_size}")
    print(f"Context length: {config.model.max_seq_len}")
    print()
    print("Configuration validation: PASS")
    print("Environment check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
