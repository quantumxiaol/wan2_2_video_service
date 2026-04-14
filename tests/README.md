
Put all your models (Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, Wan2.2-TI2V-5B) in a folder and specify the max GPU number you want to use.

```bash
bash ./tests/test.sh <local model dir> <gpu number>
```

## Device / FlashAttention tests

```bash
uv run pytest -q tests/test_device.py
uv run pytest -q tests/test_flash_attn.py
```

`tests/test_flash_attn.py` supports two benchmark profiles:

- `FLASH_ATTN_BENCH_PROFILE=advantage` (default): larger sequence lengths, usually more favorable for flash-attn.
- `FLASH_ATTN_BENCH_PROFILE=balanced`: smaller/medium sequence lengths, closer to mixed real-world scenarios.

Optional strict mode:

```bash
REQUIRE_FLASH_ATTN_SPEEDUP=1 uv run pytest -q tests/test_flash_attn.py
```

When strict mode is enabled, flash-attn must beat default SDPA in all tested shapes.
