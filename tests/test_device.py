import torch

from genpy.utils.device import get_device, get_device_info


def test_get_device_returns_valid_torch_device() -> None:
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda", "mps"}


def test_device_info_has_expected_keys() -> None:
    info = get_device_info()
    assert {"device", "device_type", "cuda_available", "cuda_device_count",
            "gpu_name", "pytorch_version", "cuda_version", "bf16_supported"} <= set(info)
