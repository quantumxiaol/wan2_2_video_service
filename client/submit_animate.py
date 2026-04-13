#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(_path=None) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a Wan-Animate generation job.")
    parser.add_argument("--image", required=True, help="Local image path.")
    parser.add_argument("--prompt", required=True, help="Generation prompt.")
    parser.add_argument(
        "--server-url",
        default=None,
        help="Server base URL. Defaults to env SERVER_URL.",
    )
    parser.add_argument("--sample-steps", type=int, default=20, help="Sampling steps.")
    parser.add_argument("--clip-len", type=int, default=77, help="Frame count (must be 4n+1).")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Polling interval in seconds.")
    parser.add_argument("--timeout", type=int, default=7200, help="Max wait seconds for job completion.")
    parser.add_argument(
        "--offload-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether server should offload some model parts to CPU during generation.",
    )
    return parser


def read_image_as_base64(image_path: Path) -> str:
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def build_server_url(cli_value: str | None) -> str:
    server_url = (cli_value or os.getenv("SERVER_URL") or "http://127.0.0.1:1111").strip()
    return server_url.rstrip("/")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    server_url = build_server_url(args.server_url)
    image_base64 = read_image_as_base64(image_path)
    create_url = f"{server_url}/v1/animate/jobs"
    payload = {
        "prompt": args.prompt,
        "image_base64": image_base64,
        "sample_steps": args.sample_steps,
        "clip_len": args.clip_len,
        "fps": args.fps,
        "seed": args.seed,
        "offload_model": args.offload_model,
    }

    with requests.Session() as session:
        create_resp = session.post(create_url, json=payload, timeout=(10, 120))
        if create_resp.status_code >= 400:
            raise SystemExit(f"Create job failed [{create_resp.status_code}]: {create_resp.text}")
        created = create_resp.json()
        job_id = created["job_id"]
        print(f"Job created: {job_id}")
        status_url = created.get("status_url", f"{server_url}/v1/animate/jobs/{job_id}")
        print(f"Status URL: {status_url}")

        started = time.time()
        while True:
            elapsed = time.time() - started
            if elapsed > args.timeout:
                raise SystemExit(f"Timed out after {args.timeout}s waiting for job {job_id}.")

            status_resp = session.get(status_url, timeout=(10, 120))
            if status_resp.status_code >= 400:
                raise SystemExit(
                    f"Query job failed [{status_resp.status_code}] for {job_id}: {status_resp.text}"
                )
            data = status_resp.json()
            status = data.get("status")
            print(f"[{int(elapsed)}s] status={status}")

            if status == "succeeded":
                print("Output video path:", data.get("output_video_path"))
                print("Output video URL:", data.get("output_video_url"))
                return
            if status == "failed":
                raise SystemExit(f"Job failed: {data.get('error')}")

            time.sleep(args.poll_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

