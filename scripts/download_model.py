#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(_path=None) -> bool:
        return False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_KEY = "t2v-A14B"
DEFAULT_HT_HOME = "./modelsweights"
CANONICAL_MODELS = ("t2v-A14B", "i2v-A14B", "ti2v-5B", "animate-14B")


def _compact(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


MODEL_ALIASES = {
    "t2v": "t2v-A14B",
    "i2v": "i2v-A14B",
    "ti2v": "ti2v-5B",
    "animate": "animate-14B",
    _compact("Wan2.2-T2V-A14B"): "t2v-A14B",
    _compact("Wan2.2-I2V-A14B"): "i2v-A14B",
    _compact("Wan2.2-TI2V-5B"): "ti2v-5B",
    _compact("Wan2.2-Animate-14B"): "animate-14B",
}
for model in CANONICAL_MODELS:
    MODEL_ALIASES[_compact(model)] = model


def normalize_model_key(raw: str) -> str:
    key = MODEL_ALIASES.get(_compact(raw.strip()))
    if not key:
        available = ", ".join(CANONICAL_MODELS)
        raise ValueError(f"Unknown model '{raw}'. Available models: {available}")
    return key


def model_arg(value: str) -> str:
    try:
        return normalize_model_key(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def resolve_ht_home(override: str | None) -> Path:
    raw = (override or os.getenv("HT_HOME") or DEFAULT_HT_HOME).strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified downloader for Wan2.2 model weights (HuggingFace or ModelScope)."
    )
    parser.add_argument(
        "--source",
        default="huggingface",
        choices=("huggingface", "modelscope"),
        help="Where to download model weights from.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_KEY,
        type=model_arg,
        help=f"Model key. Available: {', '.join(CANONICAL_MODELS)}.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all required Wan2.2 models.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision/tag.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional access token.",
    )
    parser.add_argument(
        "--mirror",
        default=None,
        help="Only for HuggingFace: optional mirror endpoint.",
    )
    parser.add_argument(
        "--ht-home",
        default=None,
        help="Override HT_HOME download root. Defaults to env HT_HOME or ./modelsweights.",
    )
    parser.add_argument(
        "--target-layout",
        default=None,
        help="Compatibility-only option. Kept for old commands.",
    )
    return parser


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    ht_home = resolve_ht_home(args.ht_home)
    ht_home.mkdir(parents=True, exist_ok=True)

    script_name = (
        "download_huggingface.py"
        if args.source == "huggingface"
        else "download_modelscope.py"
    )
    script_path = PROJECT_ROOT / "scripts" / script_name

    cmd = [sys.executable, str(script_path)]
    if args.all:
        cmd.append("--all")
    else:
        cmd.extend(["--model", args.model])
    if args.revision:
        cmd.extend(["--revision", args.revision])
    if args.token:
        cmd.extend(["--token", args.token])
    if args.mirror and args.source == "huggingface":
        cmd.extend(["--mirror", args.mirror])
    if args.ht_home:
        cmd.extend(["--ht-home", args.ht_home])
    if args.source == "modelscope" and args.target_layout:
        cmd.extend(["--target-layout", args.target_layout])

    env = os.environ.copy()
    env["HT_HOME"] = str(ht_home)
    print(f"Source: {args.source}", flush=True)
    print(f"HT_HOME: {ht_home}", flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None


if __name__ == "__main__":
    main()
