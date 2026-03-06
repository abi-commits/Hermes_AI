"""API endpoints for Hermes."""

from hermes.api.calls import router as calls_router
from hermes.api.health import router as health_router
from hermes.api.knowledge import router as knowledge_router
from hermes.api.metrics import router as metrics_router
from hermes.api.twilio import router as twilio_router

__all__ = [
    "calls_router",
    "health_router",
    "knowledge_router",
    "metrics_router",
    "twilio_router",
]
