#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import uvicorn

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(_path=None) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_server_url(raw_url: str) -> tuple[str, int]:
    normalized = raw_url.strip()
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    parsed = urlparse(normalized)
    host = parsed.hostname or "0.0.0.0"
    port = parsed.port or 1111
    return host, port


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    server_url = os.getenv("SERVER_URL", "http://0.0.0.0:1111")
    host, port = parse_server_url(server_url)
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info")
    print(f"Starting Wan-Animate server on {host}:{port}")
    uvicorn.run(
        "server.animate_service:app",
        app_dir=str(PROJECT_ROOT),
        host=host,
        port=port,
        log_level=log_level,
        timeout_keep_alive=600,
        workers=1,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
