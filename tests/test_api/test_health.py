"""Tests for health and readiness endpoints."""

import httpx
import pytest
from fastapi import FastAPI

from hermes.api.health import router


class HealthyRAG:
    """Minimal healthy RAG stub."""

    async def get_collection_stats(self) -> dict[str, int | str]:
        return {"name": "kb", "count": 1}


class UnhealthyRAG:
    """Minimal unhealthy RAG stub."""

    async def get_collection_stats(self) -> dict[str, int | str]:
        raise RuntimeError("vector db unavailable")


class HealthyTTS:
    """Minimal healthy TTS stub."""

    sample_rate = 24_000


class PingTTS:
    """TTS stub with an explicit async health check."""

    def __init__(self, healthy: bool) -> None:
        self._healthy = healthy

    async def ping(self) -> bool:
        return self._healthy


def _build_app(*, rag_service=None, tts_service=None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.rag_service = rag_service
    app.state.tts_service = tts_service
    return app


@pytest.mark.asyncio
async def test_readiness_reports_ready_when_dependencies_are_healthy():
    """Readiness should return ready only when both RAG and TTS are usable."""
    transport = httpx.ASGITransport(
        app=_build_app(rag_service=HealthyRAG(), tts_service=HealthyTTS())
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"rag": True, "tts": True},
    }


@pytest.mark.asyncio
async def test_readiness_reports_not_ready_when_dependency_checks_fail():
    """Readiness should fail closed when either dependency probe fails."""
    transport = httpx.ASGITransport(
        app=_build_app(rag_service=UnhealthyRAG(), tts_service=PingTTS(False))
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "not_ready",
        "checks": {"rag": False, "tts": False},
    }
