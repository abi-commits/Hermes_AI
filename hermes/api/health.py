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
    """Readiness probe response model."""

    status: str = Field(..., example="ready")
    checks: dict[str, bool]
    details: dict[str, str] | None = None


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
    checks: dict[str, Any] = {}
    details: dict[str, str] = {}

    # 1. RAG Check
    try:
        rag = getattr(request.app.state, "rag_service", None)
        if rag:
            stats = await rag.get_collection_stats()
            checks["rag"] = "ready"
            details["rag"] = f"Collection active with {stats.get('count', 0)} documents"
        else:
            checks["rag"] = "initializing"
            details["rag"] = "RAG service not yet attached to app state"
    except Exception as e:
        checks["rag"] = "failed"
        details["rag"] = str(e)

    # 2. TTS Check
    try:
        tts = getattr(request.app.state, "tts_service", None)
        if tts:
            if hasattr(tts, "ping"):
                if await tts.ping():
                    checks["tts"] = "ready"
                    details["tts"] = "Remote worker responsive"
                else:
                    checks["tts"] = "warming_up"
                    details["tts"] = "Remote worker not yet responding to ping"
            else:
                checks["tts"] = "ready"
                details["tts"] = "Local TTS engine initialized"
        else:
            checks["tts"] = "initializing"
            details["tts"] = "TTS service not yet attached to app state"
    except Exception as e:
        checks["tts"] = "failed"
        details["tts"] = str(e)

    # 3. LLM Check
    llm = getattr(request.app.state, "llm_service", None)
    if llm:
        checks["llm"] = "ready"
    else:
        checks["llm"] = "initializing"

    # Mandatory readiness (RAG and TTS are always required)
    is_ready = checks.get("rag") == "ready" and checks.get("tts") == "ready"
    
    # In production, LLM is also required
    settings = get_settings()
    if settings.is_production:
        is_ready = is_ready and checks.get("llm") == "ready"

    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        checks={k: v == "ready" for k, v in checks.items() if k in ["rag", "tts"]},
        details=details if not is_ready else None
    )
