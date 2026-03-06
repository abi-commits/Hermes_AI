"""Health check endpoints (liveness and readiness probes)."""

from fastapi import APIRouter, Request, status
from pydantic import BaseModel

from config import get_settings
from hermes import __version__

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    status: str
    checks: dict[str, bool]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=__version__,
        environment=settings.app_env,
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(request: Request) -> ReadinessResponse:
    """Readiness probe — checks if RAG and TTS services are initialised."""
    checks: dict[str, bool] = {}

    # Check RAG / ChromaDB connection
    try:
        rag = getattr(request.app.state, "rag_service", None)
        if rag is not None:
            checks["rag"] = True
        else:
            checks["rag"] = False
    except Exception:
        checks["rag"] = False

    # Check TTS service
    checks["tts"] = getattr(request.app.state, "tts_service", None) is not None

    all_ready = all(checks.values())

    return ReadinessResponse(
        status="ready" if all_ready else "not_ready",
        checks=checks,
    )


@router.get("/live", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def liveness_check() -> HealthResponse:
    """Liveness probe."""
    settings = get_settings()
    return HealthResponse(
        status="alive",
        version=__version__,
        environment=settings.app_env,
    )
