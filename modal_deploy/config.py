"""Configuration helpers for Hermes Modal deployments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(values)


@dataclass(frozen=True)
class ModalDeploymentConfig:
    """Local configuration used to parameterize the Modal app definition."""

    app_name: str
    endpoint_label: str
    environment: str
    cpu: float
    memory_mb: int
    concurrency: int
    timeout_s: int
    gpu: str | None
    min_containers: int
    scaledown_window_s: int
    region: str | None
    secret_names: tuple[str, ...]
    model_cache_volume_name: str | None
    model_cache_mount_path: str
    check_mode: bool
    tts_app_name: str
    tts_class_name: str
    tts_gpu: str
    tts_cpu: float
    tts_memory_mb: int
    tts_timeout_s: int
    tts_concurrency: int
    tts_min_containers: int
    tts_scaledown_window_s: int
    tts_local_workers: int
    tts_runtime_device: str


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = ModalDeploymentConfig(
    app_name=os.getenv("HERMES_MODAL_APP_NAME", "hermes-api"),
    endpoint_label=os.getenv("HERMES_MODAL_ENDPOINT_LABEL", "hermes-api"),
    environment=os.getenv("HERMES_MODAL_ENV", "production"),
    cpu=float(os.getenv("HERMES_MODAL_CPU", "2")),
    memory_mb=int(os.getenv("HERMES_MODAL_MEMORY_MB", "2048")),
    concurrency=int(os.getenv("HERMES_MODAL_CONCURRENCY", "100")),
    timeout_s=int(os.getenv("HERMES_MODAL_TIMEOUT_S", "1800")),
    gpu=os.getenv("HERMES_MODAL_GPU") or None,
    min_containers=int(os.getenv("HERMES_MODAL_MIN_CONTAINERS", "1")),
    scaledown_window_s=int(os.getenv("HERMES_MODAL_SCALEDOWN_WINDOW_S", "30")),
    region=os.getenv("HERMES_MODAL_REGION") or None,
    secret_names=_env_list("HERMES_MODAL_SECRET_NAMES") or ("hermes-prod",),
    model_cache_volume_name=os.getenv("HERMES_MODAL_MODEL_CACHE_VOLUME") or "hermes-model-cache",
    model_cache_mount_path=os.getenv("HERMES_MODAL_MODEL_CACHE_PATH", "/models"),
    check_mode=_env_flag("HERMES_MODAL_CHECK", default=False),
    tts_app_name=os.getenv("HERMES_MODAL_TTS_APP_NAME", "hermes-tts"),
    tts_class_name=os.getenv("HERMES_MODAL_TTS_CLASS_NAME", "RemoteChatterboxTTSWorker"),
    tts_gpu=os.getenv("HERMES_MODAL_TTS_GPU", "L4"),
    tts_cpu=float(os.getenv("HERMES_MODAL_TTS_CPU", "2")),
    tts_memory_mb=int(os.getenv("HERMES_MODAL_TTS_MEMORY_MB", "16384")),
    tts_timeout_s=int(os.getenv("HERMES_MODAL_TTS_TIMEOUT_S", "1800")),
    tts_concurrency=int(os.getenv("HERMES_MODAL_TTS_CONCURRENCY", "8")),
    tts_min_containers=int(os.getenv("HERMES_MODAL_TTS_MIN_CONTAINERS", "0")),
    tts_scaledown_window_s=int(os.getenv("HERMES_MODAL_TTS_SCALEDOWN_WINDOW_S", "30")),
    tts_local_workers=int(os.getenv("HERMES_MODAL_TTS_LOCAL_WORKERS", "1")),
    tts_runtime_device=os.getenv("HERMES_MODAL_TTS_RUNTIME_DEVICE", "cuda"),
)
