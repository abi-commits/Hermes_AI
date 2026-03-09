"""Health check and readiness probes for Hermes services."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, Field

from config import get_settings
from hermes import __version__

router = APIRouter()


class HealthResponse(BaseModel):
    """Liveness probe response model."""

    status: str = Field(..., example="healthy")
    version: str = Field(..., example="0.1.0")
    environment: str = Field(..., example="production")


class ReadinessResponse(BaseModel):
    """Readiness probe response model (maintained for backward compatibility)."""

    status: str = Field(..., example="ready")
    checks: dict[str, bool]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Probe",
    description="Basic check to verify the API server is running and responsive.",
)
async def health_check() -> HealthResponse:
    """Return the basic health status of the application."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=__version__,
        environment=settings.app_env,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    description="Check if all internal and external services are ready to handle traffic.",
)
async def readiness_check(request: Request) -> ReadinessResponse:
    """Check if all internal and external services are ready."""
    checks: dict[str, bool] = {}

    # 1. RAG Check
    try:
        rag = getattr(request.app.state, "rag_service", None)
        if rag:
            await rag.get_collection_stats()
            checks["rag"] = True
        else:
            checks["rag"] = False
    except Exception:
        checks["rag"] = False

    # 2. TTS Check
    try:
        tts = getattr(request.app.state, "tts_service", None)
        if tts:
            is_alive = True
            if hasattr(tts, "ping"):
                is_alive = await tts.ping()
            else:
                # Local check: ensure sample_rate is readable
                _ = tts.sample_rate
            checks["tts"] = is_alive
        else:
            checks["tts"] = False
    except Exception:
        checks["tts"] = False

    # 3. LLM Check
    llm = getattr(request.app.state, "llm_service", None)
    llm_ready = llm is not None

    # Mandatory readiness (RAG and TTS are always required)
    mandatory_ready = checks["rag"] and checks["tts"]
    
    # In production, LLM is also required
    settings = get_settings()
    if settings.is_production:
        mandatory_ready = mandatory_ready and llm_ready

    return ReadinessResponse(
        status="ready" if mandatory_ready else "not_ready",
        checks=checks,  # Only contains rag and tts
    )
