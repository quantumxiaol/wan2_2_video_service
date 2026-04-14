from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path
import warnings
from contextlib import contextmanager

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


def _strict_speedup_mode_enabled() -> bool:
    value = os.getenv("REQUIRE_FLASH_ATTN_SPEEDUP", "0").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _bench_profile() -> str:
    value = os.getenv("FLASH_ATTN_BENCH_PROFILE", "advantage").strip().lower()
    if value not in {"advantage", "balanced"}:
        raise ValueError("FLASH_ATTN_BENCH_PROFILE must be one of: advantage, balanced")
    return value


def _bench_shapes() -> list[tuple[int, int, int, int]]:
    # (batch, seq_len, heads, head_dim)
    profile = _bench_profile()
    if profile == "balanced":
        return [(2, 1536, 16, 64), (2, 2048, 16, 64)]
    # "advantage": larger sequence length where flash-attn is typically stronger.
    return [(1, 3072, 16, 64), (1, 4096, 16, 64)]


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


@contextmanager
def _sdpa_math_only():
    """
    Force SDPA to use math backend (disable flash/mem-efficient kernels).
    This provides a stable non-flash baseline for acceleration checks.
    """
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            yield
    else:
        yield


@pytest.mark.skipif(not _is_linux(), reason="flash-attn test only applies on Linux")
def test_flash_attn_available_on_linux_cuda() -> None:
    assert torch.cuda.is_available(), "CUDA is required for flash-attn tests."
    assert (
        FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE
    ), "flash-attn is not available. Install linux-gpu extra dependencies."


@pytest.mark.skipif(not _is_linux(), reason="flash-attn benchmark only applies on Linux")
def test_flash_attn_benchmark_against_sdpa(capsys: pytest.CaptureFixture[str]) -> None:
    """
    Benchmarks flash-attn against SDPA.
    - Always enforces that flash-attn is meaningfully faster than math-only SDPA
      in at least one representative shape.
    - Reports comparison against default SDPA for observability.
    - If REQUIRE_FLASH_ATTN_SPEEDUP=1, also enforces flash-attn to be faster
      than default SDPA for all tested shapes.
    """
    assert torch.cuda.is_available(), "CUDA is required for flash-attn benchmark."
    assert (
        FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE
    ), "flash-attn is not available. Install linux-gpu extra dependencies."

    device = torch.device("cuda:0")
    dtype = torch.float16
    strict_mode = _strict_speedup_mode_enabled()
    shapes = _bench_shapes()

    any_math_speedup = False
    default_sdpa_slower_cases: list[str] = []

    for batch, seq_len, heads, head_dim in shapes:
        q = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)

        def run_flash():
            with torch.no_grad():
                return flash_attention(q, k, v, causal=False, dtype=dtype)

        def run_sdpa_default():
            with torch.no_grad():
                return F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=False,
                    dropout_p=0.0,
                ).transpose(1, 2).contiguous()

        def run_sdpa_math():
            with torch.no_grad():
                with _sdpa_math_only():
                    return F.scaled_dot_product_attention(
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                        is_causal=False,
                        dropout_p=0.0,
                    ).transpose(1, 2).contiguous()

        flash_out = run_flash()
        sdpa_default_out = run_sdpa_default()
        sdpa_math_out = run_sdpa_math()
        assert flash_out.shape == sdpa_default_out.shape == sdpa_math_out.shape
        assert torch.isfinite(flash_out).all()
        assert torch.isfinite(sdpa_default_out).all()
        assert torch.isfinite(sdpa_math_out).all()

        flash_ms = _bench_cuda(run_flash)
        sdpa_default_ms = _bench_cuda(run_sdpa_default)
        sdpa_math_ms = _bench_cuda(run_sdpa_math)

        speedup_vs_default = sdpa_default_ms / flash_ms if flash_ms > 0 else float("inf")
        speedup_vs_math = sdpa_math_ms / flash_ms if flash_ms > 0 else float("inf")

        if speedup_vs_math > 1.05:
            any_math_speedup = True
        if flash_ms >= sdpa_default_ms:
            default_sdpa_slower_cases.append(
                f"B={batch},L={seq_len},H={heads},D={head_dim} "
                f"(flash={flash_ms:.3f}ms, sdpa_default={sdpa_default_ms:.3f}ms)"
            )

        with capsys.disabled():
            print(
                "[flash-attn-bench] "
                f"shape=(B={batch},L={seq_len},H={heads},D={head_dim}) "
                f"flash={flash_ms:.3f}ms "
                f"sdpa_default={sdpa_default_ms:.3f}ms "
                f"sdpa_math={sdpa_math_ms:.3f}ms "
                f"speedup_default={speedup_vs_default:.3f}x "
                f"speedup_math={speedup_vs_math:.3f}x"
            )

    assert any_math_speedup, (
        "flash-attn is available but did not beat math-only SDPA by >=5% "
        f"in tested shapes: {shapes}"
    )

    if strict_mode and default_sdpa_slower_cases:
        raise AssertionError(
            "Strict mode enabled and flash-attn was not faster than default SDPA for cases: "
            + "; ".join(default_sdpa_slower_cases)
        )
    if default_sdpa_slower_cases:
        warnings.warn(
            "flash-attn is available, but default SDPA is faster in some cases: "
            + "; ".join(default_sdpa_slower_cases),
            RuntimeWarning,
        )
