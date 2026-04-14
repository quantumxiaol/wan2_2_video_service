from __future__ import annotations

import platform

import pytest
import torch


def _platform_name() -> str:
    return platform.system().lower()


def _smoke_matmul(device: torch.device) -> None:
    x = torch.randn(512, 512, device=device)
    y = torch.randn(512, 512, device=device)
    z = (x @ y).mean()
    value = float(z.item())
    assert value == value  # NaN check


def test_device_backend_available_for_current_platform(capsys: pytest.CaptureFixture[str]) -> None:
    """
    Platform policy:
    - macOS: MPS must be available.
    - Linux: CUDA must be available.
    """
    system = _platform_name()

    if system == "darwin":
        assert torch.backends.mps.is_built(), "PyTorch MPS backend is not built."
        assert torch.backends.mps.is_available(), "MPS is not available on this macOS host."
        _smoke_matmul(torch.device("mps"))
        with capsys.disabled():
            print("[device-check] platform=darwin backend=mps device=mps")
        return

    if system == "linux":
        assert torch.cuda.is_available(), "CUDA is not available on this Linux host."
        device = torch.device("cuda:0")
        _smoke_matmul(device)
        gpu_name = torch.cuda.get_device_name(device)
        with capsys.disabled():
            print(f"[device-check] platform=linux backend=cuda device={gpu_name}")
        return

    pytest.skip(f"Unsupported platform for this test: {system}")
