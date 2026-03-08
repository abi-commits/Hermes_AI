"""Production Modal deployment wrapper for Hermes."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from pprint import pformat
from typing import Any

import modal

from modal_deploy.config import CONFIG, PROJECT_ROOT


def _build_image() -> modal.Image:
    """Build a lightweight Modal image for the production API.
    
    Uses Python 3.11 for consistency with the TTS worker and maximum stability.
    """
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            # Force specific versions to resolve ONNX/ml_dtypes compatibility issues
            "ml-dtypes>=0.5.0",
            "numpy>=1.26.0,<2.0.0",
            # Core web framework
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
    """Resolve configured Modal secrets."""
    return [modal.Secret.from_name(name) for name in CONFIG.secret_names]


def _build_volumes() -> dict[Any, Any]:
    """Create optional shared volumes for model and asset caching."""
    if not CONFIG.model_cache_volume_name:
        return {}
    return {
        CONFIG.model_cache_mount_path: modal.Volume.from_name(
            CONFIG.model_cache_volume_name,
            create_if_missing=True,
        )
    }


def _function_kwargs() -> dict[str, Any]:
    """Assemble kwargs for the Modal ASGI function."""
    kwargs: dict[str, Any] = {
        "image": _build_image(),
        "secrets": _build_secrets(),
        "cpu": CONFIG.cpu,
        "memory": CONFIG.memory_mb,
        "timeout": CONFIG.timeout_s,
        "min_containers": CONFIG.min_containers,
        "scaledown_window": CONFIG.scaledown_window_s,
        "volumes": _build_volumes(),
    }
    if CONFIG.gpu:
        kwargs["gpu"] = CONFIG.gpu
    if CONFIG.region:
        kwargs["region"] = CONFIG.region
    return kwargs


app = modal.App(CONFIG.app_name)


def _describe_config() -> dict[str, object]:
    """Return a serializable view of the active Modal deployment configuration."""
    return {
        "app_name": CONFIG.app_name,
        "endpoint_label": CONFIG.endpoint_label,
        "environment": CONFIG.environment,
        "cpu": CONFIG.cpu,
        "memory_mb": CONFIG.memory_mb,
        "concurrency": CONFIG.concurrency,
        "timeout_s": CONFIG.timeout_s,
        "gpu": CONFIG.gpu,
        "min_containers": CONFIG.min_containers,
        "scaledown_window_s": CONFIG.scaledown_window_s,
        "region": CONFIG.region,
        "secret_names": CONFIG.secret_names,
        "model_cache_volume_name": CONFIG.model_cache_volume_name,
        "model_cache_mount_path": CONFIG.model_cache_mount_path,
        "tts_app_name": CONFIG.tts_app_name,
        "tts_class_name": CONFIG.tts_class_name,
        "tts_gpu": CONFIG.tts_gpu,
    }


@app.function(**_function_kwargs())
@modal.asgi_app(label=CONFIG.endpoint_label)
@modal.concurrent(max_inputs=CONFIG.concurrency)
def hermes_api():
    """Expose Hermes as a production ASGI app on Modal."""
    os.environ.setdefault("APP_ENV", CONFIG.environment)
    os.environ.setdefault("PYTHONPATH", "/app")
    if CONFIG.model_cache_volume_name:
        os.environ.setdefault("HF_HOME", str(Path(CONFIG.model_cache_mount_path) / "hf"))

    from config import get_settings
    from hermes.main import create_app

    # Clear cached settings so the Modal-provided environment is re-read in the
    # function container before FastAPI startup.
    get_settings.cache_clear()
    return create_app()


@app.local_entrypoint()
def show_config() -> None:
    """Print the local deployment configuration before deploy/serve."""
    print(pformat(_describe_config(), sort_dicts=False))


def _run_check() -> int:
    """Validate that the Modal deployment module can be imported and configured."""
    print("Modal deployment config:")
    print(pformat(_describe_config(), sort_dicts=False))
    return 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        raise SystemExit(_run_check())
