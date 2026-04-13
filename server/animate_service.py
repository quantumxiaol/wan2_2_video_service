#!/usr/bin/env python3
from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, field_validator

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(_path=None) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def choose_target_size(width: int, height: int) -> tuple[int, int]:
    if width >= height:
        return 720, 1280
    return 1280, 720


def decode_base64_image(image_base64: str) -> Image.Image:
    payload = image_base64.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    payload = "".join(payload.split())
    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("`image_base64` is not valid base64 data.") from exc
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except OSError as exc:
        raise ValueError("Decoded bytes are not a valid image.") from exc
    return image


def letterbox(image: Image.Image, target_h: int, target_w: int) -> np.ndarray:
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Image has invalid dimensions.")
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return np.array(canvas, dtype=np.uint8)


def write_mp4_with_repeat(frame: np.ndarray, fps: int, frames: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for _ in range(frames):
            writer.append_data(frame)


def save_tensor_video(video_tensor: torch.Tensor, output_path: Path, fps: int) -> None:
    if video_tensor.ndim != 4:
        raise ValueError(f"Expected generated video tensor shape [C,T,H,W], got {tuple(video_tensor.shape)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = (
        video_tensor.detach()
        .to(dtype=torch.float32, device="cpu")
        .clamp(-1, 1)
        .permute(1, 2, 3, 0)
        .numpy()
    )
    frames = ((frames + 1.0) * 127.5).round().astype(np.uint8)
    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


@dataclass(frozen=True)
class ServiceSettings:
    results_root: Path
    outputs_root: Path
    ht_home: Path
    animate_ckpt_dir: Path
    cuda_device_id: int
    sample_solver: str

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        ht_home = resolve_path(os.getenv("HT_HOME", "./modelsweights"))
        animate_ckpt_dir = resolve_path(
            os.getenv("WAN_ANIMATE_CKPT_DIR", str(ht_home / "Wan2.2-Animate-14B"))
        )
        results_root = resolve_path(os.getenv("RESULTS_ROOT", "./results"))
        outputs_root = resolve_path(os.getenv("OUTPUTS_ROOT", "./outputs"))
        sample_solver = os.getenv("ANIMATE_SAMPLE_SOLVER", "dpm++").strip().lower()
        if sample_solver not in {"dpm++", "unipc"}:
            raise ValueError("ANIMATE_SAMPLE_SOLVER must be `dpm++` or `unipc`.")
        return cls(
            results_root=results_root,
            outputs_root=outputs_root,
            ht_home=ht_home,
            animate_ckpt_dir=animate_ckpt_dir,
            cuda_device_id=int(os.getenv("CUDA_DEVICE_ID", "0")),
            sample_solver=sample_solver,
        )


class AnimateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    image_base64: str = Field(min_length=1)
    sample_steps: int = Field(default=20, ge=4, le=80)
    clip_len: int = Field(default=77, ge=5, le=121)
    fps: int = Field(default=30, ge=1, le=60)
    seed: int | None = None
    offload_model: bool = True

    @field_validator("clip_len")
    @classmethod
    def validate_clip_len(cls, value: int) -> int:
        if (value - 1) % 4 != 0:
            raise ValueError("clip_len must satisfy 4n+1, for example: 77")
        return value


@dataclass
class JobRecord:
    job_id: str
    status: str
    prompt: str
    sample_steps: int
    clip_len: int
    fps: int
    seed: int
    offload_model: bool
    source_dir: Path
    input_image_path: Path
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    output_video_path: Path | None = None
    error: str | None = None


class WanAnimateRunner:
    def __init__(self, settings: ServiceSettings):
        self._settings = settings
        self._model: Any | None = None
        self._model_lock = threading.Lock()

    def get_model(self) -> Any:
        with self._model_lock:
            if self._model is not None:
                return self._model
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Wan-Animate service requires an NVIDIA GPU.")
            if not self._settings.animate_ckpt_dir.exists():
                raise RuntimeError(
                    f"Animate checkpoint directory does not exist: {self._settings.animate_ckpt_dir}"
                )
            from wan import WanAnimate
            from wan.configs import WAN_CONFIGS

            config = copy.deepcopy(WAN_CONFIGS["animate-14B"])
            logging.info("Loading WanAnimate model from %s", self._settings.animate_ckpt_dir)
            self._model = WanAnimate(
                config=config,
                checkpoint_dir=str(self._settings.animate_ckpt_dir),
                device_id=self._settings.cuda_device_id,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                t5_cpu=False,
                init_on_cpu=True,
                convert_model_dtype=False,
                use_relighting_lora=False,
            )
            return self._model


class AnimateJobService:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.jobs: dict[str, JobRecord] = {}
        self.jobs_lock = threading.Lock()
        self.queue: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True, name="wan-animate-worker")
        self.runner = WanAnimateRunner(settings)

        self.settings.results_root.mkdir(parents=True, exist_ok=True)
        self.settings.outputs_root.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=3)

    def create_job(self, payload: AnimateRequest) -> JobRecord:
        prompt_hash = hashlib.sha1(payload.prompt.encode("utf-8")).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"{timestamp}_{prompt_hash}_{uuid.uuid4().hex[:8]}"

        source_dir = self.settings.results_root / f"job_{job_id}"
        source_dir.mkdir(parents=True, exist_ok=True)

        image = decode_base64_image(payload.image_base64)
        input_image_path = source_dir / f"input_{job_id}.png"
        image.save(input_image_path)

        target_h, target_w = choose_target_size(*image.size)
        ref_frame = letterbox(image, target_h=target_h, target_w=target_w)
        face_frame = letterbox(image, target_h=512, target_w=512)

        Image.fromarray(ref_frame).save(source_dir / "src_ref.png")
        write_mp4_with_repeat(
            frame=ref_frame,
            fps=payload.fps,
            frames=payload.clip_len,
            output_path=source_dir / "src_pose.mp4",
        )
        write_mp4_with_repeat(
            frame=face_frame,
            fps=payload.fps,
            frames=payload.clip_len,
            output_path=source_dir / "src_face.mp4",
        )

        seed = payload.seed if payload.seed is not None else int(time.time_ns() % (2**63 - 1))
        job = JobRecord(
            job_id=job_id,
            status="queued",
            prompt=payload.prompt,
            sample_steps=payload.sample_steps,
            clip_len=payload.clip_len,
            fps=payload.fps,
            seed=seed,
            offload_model=payload.offload_model,
            source_dir=source_dir,
            input_image_path=input_image_path,
            created_at=utc_now_iso(),
        )

        with self.jobs_lock:
            self.jobs[job_id] = job

        self.queue.put(job_id)
        return job

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                job_id = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run_job(job_id)
            finally:
                self.queue.task_done()

    def _run_job(self, job_id: str) -> None:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            job.status = "running"
            job.started_at = utc_now_iso()

        try:
            model = self.runner.get_model()
            output_video_path = self.settings.outputs_root / f"wan_animate_{job.job_id}.mp4"
            video_tensor = model.generate(
                src_root_path=str(job.source_dir),
                replace_flag=False,
                clip_len=job.clip_len,
                refert_num=1,
                shift=float(model.config.sample_shift),
                sample_solver=self.settings.sample_solver,
                sampling_steps=job.sample_steps,
                guide_scale=float(model.config.sample_guide_scale),
                input_prompt=job.prompt,
                n_prompt="",
                seed=job.seed,
                offload_model=job.offload_model,
            )
            if video_tensor is None:
                raise RuntimeError("Model returned empty output tensor.")
            save_tensor_video(video_tensor, output_path=output_video_path, fps=job.fps)
            with self.jobs_lock:
                job.status = "succeeded"
                job.output_video_path = output_video_path
                job.finished_at = utc_now_iso()
        except Exception as exc:
            logging.exception("Job %s failed", job_id)
            with self.jobs_lock:
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = utc_now_iso()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def format_job_response(job: JobRecord, request: Request) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "status": job.status,
        "prompt": job.prompt,
        "sample_steps": job.sample_steps,
        "clip_len": job.clip_len,
        "fps": job.fps,
        "seed": job.seed,
        "offload_model": job.offload_model,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "source_dir": str(job.source_dir),
        "input_image_path": str(job.input_image_path),
        "error": job.error,
    }
    if job.output_video_path:
        payload["output_video_path"] = str(job.output_video_path)
        payload["output_video_url"] = f"{str(request.base_url).rstrip('/')}/outputs/{job.output_video_path.name}"
    return payload


load_dotenv(PROJECT_ROOT / ".env")
SETTINGS = ServiceSettings.from_env()
SETTINGS.outputs_root.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = AnimateJobService(SETTINGS)
    app.state.animate_service = service
    service.start()
    try:
        yield
    finally:
        service.stop()


app = FastAPI(title="Wan-Animate Service", version="0.1.0", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(SETTINGS.outputs_root)), name="outputs")


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "results_root": str(SETTINGS.results_root),
        "outputs_root": str(SETTINGS.outputs_root),
        "animate_ckpt_dir": str(SETTINGS.animate_ckpt_dir),
    }


@app.post("/v1/animate/jobs")
def create_animate_job(payload: AnimateRequest, request: Request) -> dict[str, Any]:
    service: AnimateJobService = request.app.state.animate_service
    try:
        job = service.create_job(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response = format_job_response(job, request=request)
    response["status_url"] = str(request.url_for("get_animate_job", job_id=job.job_id))
    return response


@app.get("/v1/animate/jobs/{job_id}", name="get_animate_job")
def get_animate_job(job_id: str, request: Request) -> dict[str, Any]:
    service: AnimateJobService = request.app.state.animate_service
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return format_job_response(job, request=request)
