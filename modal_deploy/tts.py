"""Dedicated Modal GPU worker for Hermes remote TTS."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import modal

from modal_deploy.config import CONFIG, PROJECT_ROOT


def _build_image() -> modal.Image:
    """Build a GPU-enabled Modal image for TTS inference.
    
    Uses Python 3.11 for maximum stability with ML dependencies.
    Includes all dependencies plus GPU-specific packages (torch, torchaudio)
    for running the Chatterbox TTS model on Modal's GPU infrastructure.
    """
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "libsndfile1",  # Audio processing library
        )
        .pip_install(
            # Resolve ONNX/ml_dtypes compatibility issues
            "ml-dtypes>=0.5.0",
            "numpy>=1.26.0,<2.0.0",
            # Core web framework (for inference server)
            "fastapi>=0.115.0",
            "uvicorn[standard]>=0.32.0",
            "python-multipart>=0.0.12",
            "httpx>=0.27.0",
            # Configuration & validation
            "pydantic>=2.9.0",
            "pydantic-settings>=2.6.0",
            # Logging & monitoring
            "structlog>=24.4.0",
            "prometheus-client>=0.21.0",
            # Data & services
            "sqlalchemy>=2.0.0",
            "chromadb>=0.5.0",
            # External services
            "tenacity>=9.0.0",
            "twilio>=9.10.0",
            "pyyaml>=6.0",
            "deepgram-sdk>=3.11.0",
            "google-genai>=1.0.0",
            # LLM & RAG
            "langchain>=0.3.0",
            "langchain-text-splitters>=0.3.0",
            "rank-bm25>=0.2.2",
            "soxr>=0.5.0",
            # GPU inference (TTS)
            "torch>=2.5.0",
            "torchaudio>=2.5.0",
            "chatterbox-streaming>=0.1.2",
            "perth>=1.0.0",
            # Modal deployment
            "modal>=1.3.5",
        )
        .workdir("/app")
        .env({"PYTHONPATH": "/app"})
        .add_local_dir(str(PROJECT_ROOT / "hermes"), "/app/hermes")
        .add_local_dir(str(PROJECT_ROOT / "config"), "/app/config")
        .add_local_dir(str(PROJECT_ROOT / "modal_deploy"), "/app/modal_deploy")
    )


def _build_secrets() -> list[modal.Secret]:
    """Resolve configured secrets for the worker."""
    return [modal.Secret.from_name(name) for name in CONFIG.secret_names]


def _build_volumes() -> dict[Any, Any]:
    """Create optional shared cache volumes."""
    if not CONFIG.model_cache_volume_name:
        return {}
    return {
        CONFIG.model_cache_mount_path: modal.Volume.from_name(
            CONFIG.model_cache_volume_name,
            create_if_missing=True,
        )
    }


app = modal.App(CONFIG.tts_app_name)


@app.cls(
    image=_build_image(),
    secrets=_build_secrets(),
    gpu=CONFIG.tts_gpu,
    cpu=CONFIG.tts_cpu,
    memory=CONFIG.tts_memory_mb,
    timeout=CONFIG.tts_timeout_s,
    scaledown_window=CONFIG.tts_scaledown_window_s,
    min_containers=CONFIG.tts_min_containers,
    volumes=_build_volumes(),
)
@modal.concurrent(max_inputs=CONFIG.tts_concurrency)
class RemoteChatterboxTTSWorker:
    """GPU-backed Chatterbox worker that returns native-rate PCM to Hermes."""

    @modal.enter()
    def enter(self) -> None:
        """Initialise the in-container TTS service once per warm worker."""
        os.environ.setdefault("PYTHONPATH", "/app")
        if CONFIG.model_cache_volume_name:
            os.environ.setdefault("HF_HOME", str(Path(CONFIG.model_cache_mount_path) / "hf"))

        from hermes.services.tts import ChatterboxTTSService

        self._executor = ThreadPoolExecutor(
            max_workers=CONFIG.tts_local_workers,
            thread_name_prefix="modal-tts",
        )
        self._service = ChatterboxTTSService(
            device=CONFIG.tts_runtime_device,
            num_workers=CONFIG.tts_local_workers,
        )
        self._service.set_executor(self._executor)

    @modal.exit()
    def exit(self) -> None:
        """Release executor resources when a worker shuts down."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    @modal.method()
    async def generate(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        embed_watermark: bool = False,
    ) -> bytes:
        """Generate full native-rate PCM audio."""
        return await self._service.generate(
            text=text,
            audio_prompt_path=audio_prompt_path,
            embed_watermark=embed_watermark,
        )

    @modal.method()
    async def generate_stream(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        embed_watermark: bool = False,
        chunk_size: int | None = None,
    ):
        """Stream native-rate PCM audio chunks."""
        async for chunk in self._service.generate_stream(
            text=text,
            audio_prompt_path=audio_prompt_path,
            embed_watermark=embed_watermark,
            chunk_size=chunk_size,
        ):
            yield chunk

    @modal.method()
    def get_sample_rate(self) -> int:
        """Expose the worker sample rate for diagnostics."""
        return self._service.sample_rate
