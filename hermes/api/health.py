"""Health check endpoints.

Provides liveness and readiness probes for Kubernetes and monitoring.
"""

from fastapi import APIRouter, status
from pydantic import BaseModel

from config import get_settings

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
    """Liveness probe - basic health check.

    Returns:
        Health status of the application.
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        environment=settings.app_env,
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """Readiness probe - check if app is ready to serve requests.

    Returns:
        Readiness status with dependency checks.
    """
    checks: dict[str, bool] = {
        "database": True,  # TODO: Check DB connection
        "redis": True,     # TODO: Check Redis connection
        "vector_db": True, # TODO: Check vector DB connection
    }

    all_ready = all(checks.values())

    return ReadinessResponse(
        status="ready" if all_ready else "not_ready",
        checks=checks,
    )


@router.get("/live", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def liveness_check() -> HealthResponse:
    """Liveness probe - check if app is running.

    Returns:
        Liveness status.
    """
    settings = get_settings()
    return HealthResponse(
        status="alive",
        version="0.1.0",
        environment=settings.app_env,
    )
