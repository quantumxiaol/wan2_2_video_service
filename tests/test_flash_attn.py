from __future__ import annotations

import importlib.util
import platform
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTENTION_PATH = PROJECT_ROOT / "wan" / "modules" / "attention.py"
_spec = importlib.util.spec_from_file_location("wan_attention_test_module", ATTENTION_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load attention module from: {ATTENTION_PATH}")
_attention_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_attention_module)

FLASH_ATTN_2_AVAILABLE = _attention_module.FLASH_ATTN_2_AVAILABLE
FLASH_ATTN_3_AVAILABLE = _attention_module.FLASH_ATTN_3_AVAILABLE
flash_attention = _attention_module.flash_attention


def _is_linux() -> bool:
    return platform.system().lower() == "linux"


def _bench_cuda(fn, *, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@pytest.mark.skipif(not _is_linux(), reason="flash-attn test only applies on Linux")
def test_flash_attn_available_on_linux_cuda() -> None:
    assert torch.cuda.is_available(), "CUDA is required for flash-attn tests."
    assert (
        FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE
    ), "flash-attn is not available. Install linux-gpu extra dependencies."


@pytest.mark.skipif(not _is_linux(), reason="flash-attn benchmark only applies on Linux")
def test_flash_attn_is_faster_than_sdpa() -> None:
    """
    Verifies flash-attn is not only importable, but also provides practical speedup
    against vanilla scaled_dot_product_attention on a representative workload.
    """
    assert torch.cuda.is_available(), "CUDA is required for flash-attn benchmark."
    assert (
        FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE
    ), "flash-attn is not available. Install linux-gpu extra dependencies."

    device = torch.device("cuda:0")
    dtype = torch.float16
    batch, seq_len, heads, head_dim = 2, 1536, 16, 64

    q = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)

    def run_flash():
        with torch.no_grad():
            out = flash_attention(q, k, v, causal=False, dtype=dtype)
            return out

    def run_sdpa():
        with torch.no_grad():
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=False,
                dropout_p=0.0,
            ).transpose(1, 2).contiguous()
            return out

    flash_out = run_flash()
    sdpa_out = run_sdpa()
    assert flash_out.shape == sdpa_out.shape
    assert torch.isfinite(flash_out).all()
    assert torch.isfinite(sdpa_out).all()

    flash_ms = _bench_cuda(run_flash)
    sdpa_ms = _bench_cuda(run_sdpa)
    # Require at least 5% speedup to claim "really accelerating".
    assert flash_ms < sdpa_ms * 0.95, (
        f"flash-attn is not faster enough: flash={flash_ms:.3f}ms, sdpa={sdpa_ms:.3f}ms"
    )
