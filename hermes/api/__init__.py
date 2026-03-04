"""API endpoints for Hermes."""

from hermes.api.health import router as health_router
from hermes.api.metrics import router as metrics_router

__all__ = ["health_router", "metrics_router"]
