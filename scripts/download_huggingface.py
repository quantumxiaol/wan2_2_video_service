#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(_path=None) -> bool:
        return False
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_KEY = "t2v-A14B"
DEFAULT_HT_HOME = "./modelsweights"

MODEL_SPECS: dict[str, dict[str, str]] = {
    "t2v-A14B": {
        "repo_id": "Wan-AI/Wan2.2-T2V-A14B",
        "local_dir_name": "Wan2.2-T2V-A14B",
    },
    "i2v-A14B": {
        "repo_id": "Wan-AI/Wan2.2-I2V-A14B",
        "local_dir_name": "Wan2.2-I2V-A14B",
    },
    "ti2v-5B": {
        "repo_id": "Wan-AI/Wan2.2-TI2V-5B",
        "local_dir_name": "Wan2.2-TI2V-5B",
    },
    "animate-14B": {
        "repo_id": "Wan-AI/Wan2.2-Animate-14B",
        "local_dir_name": "Wan2.2-Animate-14B",
    },
}


def _compact(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _build_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {
        "t2v": "t2v-A14B",
        "i2v": "i2v-A14B",
        "ti2v": "ti2v-5B",
        "animate": "animate-14B",
    }
    for canonical, spec in MODEL_SPECS.items():
        candidates = {
            canonical,
            spec["local_dir_name"],
            spec["repo_id"],
            spec["repo_id"].split("/")[-1],
        }
        for candidate in candidates:
            aliases[_compact(candidate)] = canonical
    return aliases


MODEL_ALIASES = _build_aliases()


def normalize_model_key(raw: str) -> str:
    model_key = MODEL_ALIASES.get(_compact(raw.strip()))
    if not model_key:
        available = ", ".join(MODEL_SPECS.keys())
        raise ValueError(f"Unknown model '{raw}'. Available models: {available}")
    return model_key


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


def configure_hf_endpoint(mirror: str | None) -> str | None:
    if mirror:
        os.environ["HF_ENDPOINT"] = mirror
    elif os.getenv("HF_MIRROR") and not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = os.getenv("HF_MIRROR", "")
    return os.getenv("HF_ENDPOINT")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Wan2.2 models from HuggingFace.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_KEY,
        type=model_arg,
        help=f"Model key. Available: {', '.join(MODEL_SPECS.keys())}.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all Wan2.2 models required by this project.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision/branch/tag.",
    )
    parser.add_argument(
        "--mirror",
        default=None,
        help="Optional mirror endpoint, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HuggingFace token.",
    )
    parser.add_argument(
        "--ht-home",
        default=None,
        help="Override HT_HOME download root. Defaults to env HT_HOME or ./modelsweights.",
    )
    return parser


def download_one(
    model_key: str,
    ht_home: Path,
    revision: str | None,
    token: str | None,
    endpoint: str | None,
) -> str:
    spec = MODEL_SPECS[model_key]
    target_dir = ht_home / spec["local_dir_name"]
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "repo_id": spec["repo_id"],
        "local_dir": str(target_dir),
        "cache_dir": str(ht_home / ".cache" / "huggingface"),
    }
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token
    if endpoint:
        kwargs["endpoint"] = endpoint

    return snapshot_download(**kwargs)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    endpoint = configure_hf_endpoint(args.mirror)
    ht_home = resolve_ht_home(args.ht_home)
    ht_home.mkdir(parents=True, exist_ok=True)
    os.environ["HT_HOME"] = str(ht_home)

    if args.all:
        model_keys = sorted(MODEL_SPECS)
    else:
        model_keys = [args.model]

    print(f"HT_HOME: {ht_home}")
    if endpoint:
        print(f"HF endpoint: {endpoint}")

    for key in model_keys:
        repo_id = MODEL_SPECS[key]["repo_id"]
        print(f"Downloading {key} from HuggingFace repo {repo_id} ...")
        path = download_one(
            key,
            ht_home=ht_home,
            revision=args.revision,
            token=args.token,
            endpoint=endpoint,
        )
        print(f"Done: {path}")


if __name__ == "__main__":
    main()
