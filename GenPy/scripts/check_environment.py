"""Report the local environment relevant to future GenPy training."""

import platform

import torch

try:
    from genpy.utils.device import get_device
except ModuleNotFoundError:  # Allows the required direct script invocation.
    def get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bf16_supported() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        return False


def _fp16_available() -> bool:
    return bool(torch.cuda.is_available())


def main() -> int:
    cuda_available = bool(torch.cuda.is_available())
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"PyTorch CUDA version: {torch.version.cuda or 'None'}")
    print(f"cuDNN version: {torch.backends.cudnn.version() or 'None'}")
    device_count = torch.cuda.device_count() if cuda_available else 0
    print(f"CUDA device count: {device_count}")
    if cuda_available:
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            memory_gib = properties.total_memory / (1024 ** 3)
            print(f"GPU {index} name: {properties.name}")
            print(f"GPU {index} VRAM: {memory_gib:.2f} GiB")
    else:
        print("GPU name: None")
        print("GPU VRAM: None")
    print(f"BF16 support: {_bf16_supported()}")
    print(f"FP16 CUDA training available: {_fp16_available()}")
    print(f"Selected GenPy device: {get_device()}")
    if not cuda_available:
        print("CUDA GPU not detected.")
        print("CPU development mode is available.")
        print("Use Kaggle GPU for GenPy pretraining.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
